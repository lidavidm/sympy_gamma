from .utils import mathjax_latex

import sympy
from sympy import latex


class ResultCardMeta(type):
    cards = {}

    def __new__(meta, name, bases, dct):
        klass = super(ResultCardMeta, meta).__new__(meta, name, bases, dct)
        ResultCardMeta.cards[dct['name']] = klass
        return klass


class ResultCard(object):
    """
    Represents a potential output card.
    """

    __metaclass__ = ResultCardMeta

    name = 'defaultcard'
    title = '(unnamed result)'
    result_statement = ''
    multivariate = True
    parameters = {}

    def __init__(self, *arguments):
        """
        arguments - Specifies which parts of 'components' this result card
        should operate on.
        """
        if not arguments:
            arguments = ['input_evaluated']

        self.arguments = arguments

    def eval(self, evaluator, components, parameters=None):
        """
        Evaluates this card and returns the unformatted Python object result.
        """
        assert len(self.arguments) >= 1

        parameters = self.default_parameters(parameters)
        return self.eval_method(evaluator, components, parameters, self.arguments[0])

    def eval_method(self, evaluator, components, parameters, argument):
        """
        Intended to be overridden to customize execution.
        """
        params = parameters.copy()
        params.update(components)
        line = self.result_statement.format(_var=components['variable'], **params)
        return evaluator.eval(line, use_none_for_exceptions=True,
                              repr_expression=False)

    def title_text(self, evaluated, components):
        return self.title

    def input_text(self, evaluated, components):
        return self.result_statement.format(_var=components['variable'], **components)

    def pre_output_latex(self, evaluated, components):
        pass

    def output_latex(self, output):
        return mathjax_latex(output)

    def output_text(self, output):
        pass

    def default_parameters(self, params):
        if params is None:
            params = {}
        for arg in self.parameters:
            params.setdefault(arg, self.parameters[arg])

        return params

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.title == other.title and
                self.result_statement == other.result_statement and
                self.multivariate == other.multivariate and
                self.arguments == other.arguments)

    def __repr__(self):
        return "<ResultCard '{}'>".format(self.title)


class Roots(ResultCard):
    name = 'roots'
    title = "Roots"
    result_statement = "solve({input_evaluated}, {_var})"

    def pre_output_latex(self, evaluated, components):
        return latex(components['variable'])


class Diff(ResultCard):
    name = 'diff'
    title = "Derivative"
    result_statement = "diff({input_evaluated}, {_var})"

    def pre_output_latex(self, evaluated, components):
        return latex(sympy.Derivative(evaluated))


class Series(ResultCard):
    name = 'series'
    title = "Series expansion around 0, first 10 terms"
    result_statement = "series({input_evaluated}, {_var}, 0, 10)"

    def pre_output_latex(self, evaluated, components):
        return ''

class Plot(ResultCard):
    name = 'plot'
    title = "Plot"

    def eval_method(self, evaluator, components, parameters, argument):
        return repr(parameters)

def get_card(name):
    return ResultCardMeta.cards[name]
