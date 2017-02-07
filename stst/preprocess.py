# coding: utf8
from __future__ import print_function

from enchant.checker import SpellChecker
from enchant.checker.CmdLineChecker import CmdLineChecker


def contraction_replacement(line):
    """
    中文编码 变化？
    wouldn't -> would not
    :param line:
    :return:
    """
    pass


def remove_URLs(line):
    pass


def abbriviation_normalization(line):
    pass


def negation_replacement(line):
    pass


chkr = SpellChecker("en_US")
cmdln = CmdLineChecker()


def misspelling_correction(line):
    chkr.set_text(line)
    cmdln.set_checker(chkr)
    cmdln.run()
    return chkr.get_text()


# def remove_stopwords(line):
#    pass

# def lemmatization(line):
#    pass

misspelling_correction("this is sme example txt")
