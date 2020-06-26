import inspect
import sys, os
import yaml

"""
Provides decorators for automatically generating the boilerplate for
writing stepfiles in python.

Python stepfiles should decorate function with:
@step(prereqs=..., tags=...)

The last thing a python stepfile should then do is:
steps.run(locals())

For convenience, also exposes basedir for finding the root of the datafoundry
checkout in python.
"""

basedir = os.path.split(__file__)
basedir = os.path.split(basedir[0])[0]  # YES! Doing this twice.


def step(tags=None, prereqs=None, undo=None, tests=None):
    """attach stepinfo to a callable, making it a step"""
    def fn(oldfn):
        oldfn.stepinfo = {}
        if tags is not None:
            oldfn.stepinfo['tags'] = tags
        if prereqs is not None:
            oldfn.stepinfo['prereqs'] = prereqs
        if undo is not None:
            oldfn.stepinfo['undo'] = undo
        if tests is not None:
            oldfn.stepinfo['tests'] = tests
        return oldfn
    return fn

def reflect_spec(fn):
    """look at one value to see if it is a callable with stepinfo (a "step")"""
    if not hasattr(fn, 'stepinfo'):
        return None
    fnspec = fn.stepinfo
    args, varargs, keywords, defaults = inspect.getargspec(fn)
    assert not varargs
    assert not keywords
    fnspec['args'] = list(args)
    if defaults:
        fnspec['defaults'] = {}
        for default_i, default in enumerate(defaults or []):
            i = len(args) - len(defaults) + default_i
            fnspec['defaults'][args[i]] = default
    return fnspec

def reflect_all(scope):
    """extract all specs from the top level values in scope"""
    spec = {}
    for fn in scope.values():
        fnspec = reflect_spec(fn)
        if fnspec:
            spec[fn] = fnspec
    return spec

def run(*scopes):
    """reflect dicts containing stepinfo annotated callables ('steps'); merge them from left to right
       (so rightmost defs take precedence) and implement the commandline protocol"""
    scope = {}
    for contributor in scopes:
        scope.update(contributor)
    verb = len(sys.argv) > 1 and sys.argv[1]
    args = len(sys.argv) > 2 and sys.argv[2:] or []
    if not verb or verb == 'help' or verb not in scope:
        for fn, spec in reflect_all(scope).items():
            if fn.__doc__:
                print "\n".join("# " + x.strip() for x in fn.__doc__.splitlines())
            print yaml.dump({fn.func_name: spec})
    else:
        scope[verb](*args)
