import sys
import json
import itertools
import sympy
from sympy.core.function import FunctionClass
from sympy.core.symbol import Symbol
import docutils.core
import diffsteps
import intsteps


from .resultcards import *


# Decide which result card set to use

def is_derivative(input_evaluated):
    return isinstance(input_evaluated, sympy.Derivative)

def is_integral(input_evaluated):
    return isinstance(input_evaluated, sympy.Integral)

def is_integer(input_evaluated):
    return isinstance(input_evaluated, sympy.Integer)

def is_rational(input_evaluated):
    return isinstance(input_evaluated, sympy.Rational) and not input_evaluated.is_Integer

def is_float(input_evaluated):
    return isinstance(input_evaluated, sympy.Float)

def is_numbersymbol(input_evaluated):
    return isinstance(input_evaluated, sympy.NumberSymbol)

def is_constant(input_evaluated):
    # is_constant reduces trig identities (even with simplify=False?) so we
    # check free_symbols instead
    return (hasattr(input_evaluated, 'free_symbols') and
            not input_evaluated.free_symbols)

def is_approximatable_constant(input_evaluated):
    # is_constant, but exclude Integer/Float/infinity
    return (hasattr(input_evaluated, 'free_symbols') and
            not input_evaluated.free_symbols and
            not input_evaluated.is_Integer and
            not input_evaluated.is_Float and
            input_evaluated.is_finite is not True)

def is_complex(input_evaluated):
    try:
        return sympy.I in input_evaluated.atoms()
    except (AttributeError, TypeError):
        return False

def is_trig(input_evaluated):
    try:
        if (isinstance(input_evaluated, sympy.Basic) and
            any(input_evaluated.find(func)
                for func in (sympy.sin, sympy.cos, sympy.tan,
                             sympy.csc, sympy.sec, sympy.cot))):
            return True
    except AttributeError:
        pass
    return False

def is_not_constant_basic(input_evaluated):
    return (not is_constant(input_evaluated) and
            isinstance(input_evaluated, sympy.Basic) and
            not is_logic(input_evaluated))

def is_uncalled_function(input_evaluated):
    return hasattr(input_evaluated, '__call__') and not isinstance(input_evaluated, sympy.Basic)

def is_matrix(input_evaluated):
    return isinstance(input_evaluated, sympy.Matrix)

def is_logic(input_evaluated):
    return isinstance(input_evaluated, (sympy.And, sympy.Or, sympy.Not, sympy.Xor))

def is_sum(input_evaluated):
    return isinstance(input_evaluated, sympy.Sum)

def is_product(input_evaluated):
    return isinstance(input_evaluated, sympy.Product)


# Functions to convert input and extract variable used

def default_variable(arguments, evaluated):
    try:
        variables = list(evaluated.atoms(sympy.Symbol))
    except:
        variables = []

    return {
        'variables': variables,
        'variable': variables[0] if variables else None,
        'input_evaluated': evaluated
    }

def extract_first(arguments, evaluated):
    result = default_variable(arguments, evaluated)
    result['input_evaluated'] = arguments[1][0]
    return result

def extract_integral(arguments, evaluated):
    limits = arguments[1][1:]
    variables = []

    if not limits:
        variables = [arguments[1][0].atoms(sympy.Symbol).pop()]
        limits = variables
    else:
        for limit in limits:
            if isinstance(limit, tuple):
                variables.append(limit[0])
            else:
                variables.append(limit)

    return {
        'integrand': arguments[1][0],
        'variables': variables,
        'variable': variables[0],
        'limits': limits
    }

def extract_derivative(arguments, evaluated):
    variables = list(sorted(arguments[1][0].atoms(sympy.Symbol), key=lambda x: x.name))

    variable = arguments[1][1:]
    if variable:
        variables.remove(variable[0])
        variables.insert(0, variable[0])

    return {
        'function': arguments[1][0],
        'derivative': evaluated,
        'variables': variables,
        'variable': variables[0],
        'input_evaluated': arguments[1][0]
    }

def extract_plot(arguments, evaluated):
    result = {}
    if arguments.args:
        if isinstance(arguments.args[0], sympy.Basic):
            result['variables'] = list(arguments.args[0].atoms(sympy.Symbol))
            result['variable'] = result['variables'][0]
            result['input_evaluated'] = [arguments.args[0]]

            if len(result['variables']) != 1:
                raise ValueError("Cannot plot function of multiple variables")
        else:
            variables = set()
            try:
                for func in arguments.args[0]:
                    variables.update(func.atoms(sympy.Symbol))
            except TypeError:
                raise ValueError("plot() accepts either one function, a list of functions, or keyword arguments")

            variables = list(variables)
            if len(variables) > 1:
                raise ValueError('All functions must have the same and at most one variable')
            if len(variables) == 0:
                variables.append(sympy.Symbol('x'))
            result['variables'] = variables
            result['variable'] = variables[0]
            result['input_evaluated'] = arguments.args[0]
    elif arguments.kwargs:
        result['variables'] = [sympy.Symbol('x')]
        result['variable'] = sympy.Symbol('x')

        parametrics = 1
        functions = {}
        for f in arguments.kwargs:
            if f.startswith('x'):
                y_key = 'y' + f[1:]
                if y_key in arguments.kwargs:
                    # Parametric
                    x = arguments.kwargs[f]
                    y = arguments.kwargs[y_key]
                    functions['p' + str(parametrics)] = (x, y)
                    parametrics += 1
            else:
                if f.startswith('y') and ('x' + f[1:]) in arguments.kwargs:
                    continue
                functions[f] = arguments.kwargs[f]
        result['input_evaluated'] = functions
    return result

# Formatting functions

_function_formatters = {}
def formats_function(name):
    def _formats_function(func):
        _function_formatters[name] = func
        return func
    return _formats_function

@formats_function('diophantine')
def format_diophantine(result, arguments, formatter):
    variables = list(sorted(arguments.args[0].atoms(sympy.Symbol), key=str))
    if isinstance(result, set):
        return format_nested_list_title(*variables)(result, formatter)
    else:
        return format_nested_list_title(*variables)([result], formatter)

def format_by_type(result, arguments=None, formatter=None, function_name=None):
    """
    Format something based on its type and on the input to Gamma.
    """
    if arguments and not function_name:
        function_name = arguments[0]
    if function_name in _function_formatters:
        return _function_formatters[function_name](result, arguments, formatter)
    elif function_name in all_cards and 'format_output_function' in all_cards[function_name].card_info:
        return all_cards[function_name].format_output(result, formatter)
    elif isinstance(result, (list, tuple)):
        return format_list(result, formatter)
    else:
        return formatter(result)

def format_nothing(arg, formatter):
    return arg

def format_steps(arg, formatter):
    return '<div class="steps">{}</div>'.format(arg)

def format_long_integer(line, integer, variable):
    intstr = str(integer)
    if len(intstr) > 100:
        # \xe2 is Unicode ellipsis
        return intstr[:20] + "..." + intstr[len(intstr) - 21:]
    return line % intstr

def format_integral(line, result, components):
    if components['limits']:
        limits = ', '.join(map(repr, components['limits']))
    else:
        limits = ', '.join(map(repr, components['variables']))

    return line.format(_var=limits) % components['integrand']

def format_function_docs_input(line, function, components):
    function = getattr(components['input_evaluated'], '__name__', str(function))
    return line % function

def format_dict_title(*title):
    def _format_dict(dictionary, formatter):
        html = ['<table>',
                '<thead><tr><th>{}</th><th>{}</th></tr></thead>'.format(*title),
                '<tbody>']
        try:
            fdict = dictionary.iteritems()
            if not any(isinstance(i,Symbol) for i in dictionary.keys()):
                fdict = sorted(dictionary.iteritems())
            for key, val in fdict:
                html.append('<tr><td>{}</td><td>{}</td></tr>'.format(key, val))
        except AttributeError, TypeError:  # not iterable/not a dict
            return formatter(dictionary)
        html.append('</tbody></table>')
        return '\n'.join(html)
    return _format_dict

def format_list(items, formatter):
    try:
        if len(items) == 0:
            return "<p>No result</p>"
        html = ['<ul>']
        for item in items:
            html.append('<li>{}</li>'.format(formatter(item)))
        html.append('</ul>')
        return '\n'.join(html)
    except TypeError:  # not iterable, like None
        return formatter(items)

def format_nested_list_title(*titles):
    def _format_nested_list_title(items, formatter):
        try:
            if len(items) == 0:
                return "<p>No result</p>"
            html = ['<table>', '<thead><tr>']
            for title in titles:
                html.append('<th>{}</th>'.format(title))
            html.append('</tr></thead>')
            html.append('<tbody>')
            for item in items:
                html.append('<tr>')
                for subitem in item:
                    html.append('<td>{}</td>'.format(formatter(subitem)))
                html.append('</tr>')
            html.append('</tbody></table>')
            return '\n'.join(html)
        except TypeError:  # not iterable, like None
            return formatter(items)
    return _format_nested_list_title

def format_series_fake_title(title, evaluated):
    if len(evaluated.args) >= 3:
        about = evaluated.args[2]
    else:
        about = 0
    if len(evaluated.args) >= 4:
        up_to = evaluated.args[3]
    else:
        up_to = 6
    return title.format(about, up_to)

def format_truth_table(table, formatter):
    # table is (variables, [(bool, bool...)] representing combination of values
    # and result
    variables, table = table
    titles = list(map(str, variables))
    titles.append("Value")
    def formatter(x):
        if x is True:
            return '<span class="true">True</span>'
        elif x is False:
            return '<span class="false">False</span>'
        else:
            return str(x)
    return format_nested_list_title(*titles)(table, formatter)

def format_approximator(approximation, formatter):
    obj, digits = approximation
    return formatter(obj, r'\approx', obj.evalf(digits))

DIAGRAM_CODE = """
<div class="factorization-diagram" data-primes="{primes}">
    <div></div>
    <p><a href="http://mathlesstraveled.com/2012/10/05/factorization-diagrams/">About this diagram</a></p>
</div>
"""

def format_factorization_diagram(factors, formatter):
    primes = []
    for prime in reversed(sorted(factors)):
        times = factors[prime]
        primes.extend([prime] * times)
    return DIAGRAM_CODE.format(primes=primes)

PLOTTING_CODE = """
<div class="plot"
     data-variable="{variable}">
<div class="graphs">{graphs}</div>
</div>
"""

def format_plot(plot_data, formatter):
    return PLOTTING_CODE.format(**plot_data)

def format_plot_input(result_statement, input_repr, components):
    if 'input_evaluated' in components:
        functions = components['input_evaluated']
        if isinstance(functions, list):
            functions = ['<span>{}</span>'.format(f) for f in functions]
            if len(functions) > 1:
                return 'plot([{}])'.format(', '.join(functions))
            else:
                return 'plot({})'.format(functions[0])
        elif isinstance(functions, dict):
            return 'plot({})'.format(', '.join(
                '<span>{}={}</span>'.format(y, x)
                for y, x in functions.items()))
    else:
        return 'plot({})'.format(input_repr)

GRAPH_TYPES = {
    'xy': [lambda x, y: x, lambda x, y: y],
    'parametric': [lambda x, y: x, lambda x, y: y],
    'polar': [lambda x, y: float(y * sympy.cos(x)),
              lambda x, y: float(y * sympy.sin(x))]
}

def determine_graph_type(key):
    if key.startswith('r'):
        return 'polar'
    elif key.startswith('p'):
        return 'parametric'
    else:
        return 'xy'

def eval_plot(evaluator, components, parameters=None):
    if parameters is None:
        parameters = {}

    xmin, xmax = parameters.get('xmin', -10), parameters.get('xmax', 10)
    pmin, pmax = parameters.get('tmin', 0), parameters.get('tmax', 2 * sympy.pi)
    tmin, tmax = parameters.get('tmin', 0), parameters.get('tmax', 10)
    from sympy.plotting.plot import LineOver1DRangeSeries, Parametric2DLineSeries
    functions = evaluator.get("input_evaluated")
    if isinstance(functions, sympy.Basic):
        functions = [(functions, 'xy')]
    elif isinstance(functions, list):
        functions = [(f, 'xy') for f in functions]
    elif isinstance(functions, dict):
        functions = [(f, determine_graph_type(key)) for key, f in functions.items()]

    graphs = []
    for func, graph_type in functions:
        if graph_type == 'parametric':
            x_func, y_func = func
            x_vars, y_vars = x_func.free_symbols, y_func.free_symbols
            variables = x_vars.union(y_vars)
            if x_vars != y_vars:
                raise ValueError("Both functions in a parametric plot must have the same variable")
        else:
            variables = func.free_symbols

        if len(variables) > 1:
            raise ValueError("Cannot plot multivariate function")
        elif len(variables) == 0:
            variable = sympy.Symbol('x')
        else:
            variable = list(variables)[0]

        try:
            if graph_type == 'xy':
                graph_range = (variable, xmin, xmax)
            elif graph_type == 'polar':
                graph_range = (variable, pmin, pmax)
            elif graph_type == 'parametric':
                graph_range = (variable, tmin, tmax)

            if graph_type in ('xy', 'polar'):
                series = LineOver1DRangeSeries(func, graph_range, nb_of_points=150)
            elif graph_type == 'parametric':
                series = Parametric2DLineSeries(x_func, y_func, graph_range, nb_of_points=150)
            # returns a list of [[x,y], [next_x, next_y]] pairs
            series = series.get_segments()
        except TypeError:
            raise ValueError("Cannot plot function")

        xvalues = []
        yvalues = []

        def limit_y(y):
            CEILING = 1e8
            if y > CEILING:
                y = CEILING
            if y < -CEILING:
                y = -CEILING
            return y

        x_transform, y_transform = GRAPH_TYPES[graph_type]
        series.append([series[-1][1], None])
        for point in series:
            if point[0][1] is None:
                continue
            x = point[0][0]
            y = limit_y(point[0][1])
            xvalues.append(x_transform(x, y))
            yvalues.append(y_transform(x, y))

        graphs.append({
            'type': graph_type,
            'function': sympy.jscode(sympy.sympify(func)),
            'points': {
                'x': xvalues,
                'y': yvalues
            },
            'data': None
        })
    return {
        'variable': repr(variable),
        'graphs': json.dumps(graphs)
    }

def eval_factorization(evaluator, components, parameters=None):
    number = evaluator.get("input_evaluated")

    if number == 0:
        raise ValueError("Can't factor 0")

    factors = sympy.ntheory.factorint(number, limit=100)
    smallfactors = {}
    for factor in factors:
        if factor <= 100:
            smallfactors[factor] = factors[factor]
    return smallfactors

def eval_factorization_diagram(evaluator, components, parameters=None):
    # Raises ValueError (stops card from appearing) if the factors are too
    # large so that the diagram will look nice
    number = int(evaluator.eval("input_evaluated"))
    if number > 256:
        raise ValueError("Number too large")
    elif number == 0:
        raise ValueError("Can't factor 0")
    factors = sympy.ntheory.factorint(number, limit=101)
    smallfactors = {}
    for factor in factors:
        if factor <= 256:
            smallfactors[factor] = factors[factor]
        else:
            raise ValueError("Number too large")
    return smallfactors

def eval_integral(evaluator, components, parameters=None):
    return sympy.integrate(components['integrand'], *components['limits'])

def eval_integral_manual(evaluator, components, parameters=None):
    return sympy.integrals.manualintegrate(components['integrand'],
                                           components['variable'])

def eval_diffsteps(evaluator, components, parameters=None):
    function = components.get('function', evaluator.get('input_evaluated'))

    return diffsteps.print_html_steps(function,
                                      components['variable'])

def eval_intsteps(evaluator, components, parameters=None):
    integrand = components.get('integrand', evaluator.get('input_evaluated'))

    return intsteps.print_html_steps(integrand, components['variable'])

# http://www.python.org/dev/peps/pep-0257/
def trim(docstring):
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxint
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxint:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)

def eval_function_docs(evaluator, components, parameters=None):
    docstring = trim(evaluator.get("input_evaluated").__doc__)
    return docutils.core.publish_parts(docstring, writer_name='html4css1',
                                       settings_overrides={'_disable_config': True})['html_body']

def eval_truth_table(evaluator, components, parameters=None):
    expr = evaluator.get("input_evaluated")
    variables = list(sorted(expr.atoms(sympy.Symbol), key=str))

    result = []
    for combination in itertools.product([True, False], repeat=len(variables)):
        result.append(combination +(expr.subs(zip(variables, combination)),))
    return variables, result



def eval_approximator(evaluator, components, parameters=None):
    if parameters is None:
        raise ValueError
    digits = parameters.get('digits', 10)
    return (evaluator.get('input_evaluated'), digits)

# Result cards

"""
Syntax:

(predicate, extract_components, result_cards)

predicate: str or func
  If a string, names a function that uses this set of result cards.
  If a function, the function, given the evaluated input, returns True if
  this set of result cards should be used.

extract_components: None or func
  If None, use the default function.
  If a function, specifies a function that parses the input expression into
  a components dictionary. For instance, for an integral, this function
  might extract the limits, integrand, and variable.

result_cards: None or list
  If None, do not show any result cards for this function beyond the
  automatically generated 'Result' and 'Simplification' cards (if they are
  applicable).
  If a list, specifies a list of result cards to display.
"""

result_sets = [
    ('diff', extract_derivative, [Diff(), Plot('input_evaluated', 'derivative')]),
    # ('diff', extract_derivative, [Diff(), DiffSteps(), Plot('input_evaluated', 'derivative')]),
    (is_not_constant_basic, None, [Roots(), Diff(), Series()])
]

learn_more_sets = {
    'rsolve': ['http://en.wikipedia.org/wiki/Recurrence_relation',
               'http://mathworld.wolfram.com/RecurrenceEquation.html',
               'http://docs.sympy.org/latest/modules/solvers/solvers.html#recurrence-equtions']
}

def is_function_handled(function_name):
    """Do any of the result sets handle this specific function?"""
    if function_name == "simplify":
        return True
    return any(name == function_name for (name, _, cards) in result_sets if cards is not None)

def find_result_set(function_name, input_evaluated):
    """
    Finds a set of result cards based on function name and evaluated input.

    Returns:

    - Function that parses the evaluated input into components. For instance,
      for an integral this would extract the integrand and limits of integration.
      This function will always extract the variables.
    - List of result cards.
    """
    result = []
    result_converter = default_variable

    for predicate, converter, result_cards in result_sets:
        if predicate == function_name:
            if converter:
                result_converter = converter
            if result_cards is None:
                return result_converter, result
            for card in result_cards:
                if card not in result:
                    result.append(card)
        elif callable(predicate) and predicate(input_evaluated):
            if converter:
                result_converter = converter
            if result_cards is None:
                return result_converter, result
            for card in result_cards:
                if card not in result:
                    result.append(card)

    return result_converter, result

def find_learn_more_set(function_name):
    urls = learn_more_sets.get(function_name)
    if urls:
        return '<div class="document"><ul>{}</ul></div>'.format('\n'.join('<li><a href="{0}">{0}</a></li>'.format(url) for url in urls))
