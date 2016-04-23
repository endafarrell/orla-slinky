import sys
import os
import yaml
from yaml.scanner import ScannerError
import subprocess
import time, datetime
from glob import glob
from fnmatch import fnmatch
import functools
from collections import defaultdict
import getpass
# try:
#     import pysvn
# except ImportError, x:
#     sys.stderr.write("WARNING: cannot import pysvn in {}\n".format(__file__))
#     sys.stderr.flush()
import eutils

sys.stderr.write("WARNING: cannot import pysvn in {}\n".format(__file__))
sys.stderr.flush()
import json

from steps import basedir
from eutils import pipetee
#from load_monitor import LoadMonitor

# use for evil past python stepfile reading
import imp
from StringIO import StringIO
import traceback

from string import Formatter


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


class Tag(object):
    """ducktyped standin for Step to represent a tag"""
    def __init__(self, name, prereqs):
        self.tag = True
        self.stepfile = "(tag)"
        self.name = name
        self.verb = None
        self.args = []
        self.tags = []
        self.tests = []
        self.prereqs = prereqs # Slinky computes the actual prereqs for the tag
        self.consumers = [] # will be filled in by Slinky


def make_list(s):
    """turn a single string into a one element list; turn None into []; leave lists as is"""
    if isinstance(s, str):
        return [s]
    return s or []


def log_file_time_stamp():
    return str(int(time.time()*1000))


class Step(object):
    """an individual step inside a Stepfile"""
    def __init__(self, stepfile, name, info):
        self.tag = False
        self.stepfile = stepfile
        self.name = self.stepfile.module + "." + name
        self.verb = name
        self.consumers = []  # will be filled in by Slinky

        self.args = make_list(info.get('args'))
        self.tags = make_list(info.get('tags'))
        self.prereqs = make_list(info.get('prereqs'))
        self.tests = make_list(info.get('tests'))
        assert len(self.tests) <= 1 # not sure sane behavior yet for >1 entry here


        if self.prereqs:
            # if prereq starts with a dot, prepend the local module name
            self.prereqs = [x.startswith('.') and "%s%s"%(self.stepfile.module, x) or x for x in self.prereqs]
        self.defaults = info.get('defaults') or {}

        self.undo = info.get('undo', None) # either a single step, or None; may not be a list
        if self.undo and self.undo.startswith('.'):
            self.undo = self.stepfile.module + self.undo

    def statusfile(self):
        return os.path.join(self.stepfile.slinky.statedir, self.name)

    def logfile(self):
        return os.path.join(self.stepfile.slinky.logdir, "%s.%s" % (self.name, log_file_time_stamp()))

    def runfile(self):
        return os.path.join(self.stepfile.slinky.rundir, self.name)

    def failfile(self):
        return os.path.join(self.stepfile.slinky.faildir, self.name)

    def fakedone(self):
        with open(self.statusfile(), 'w') as status:
            status.write("done")

    def cleardone(self):
        f = self.statusfile()
        if os.path.isfile(f):
            os.unlink(f)

    def isdone(self):
        return os.path.isfile(self.statusfile())

    def _getargs(self, config):
        args = []
        for arg in self.args:
            if arg in config:
                value = config[arg]
            elif arg in self.defaults:
                value = self.defaults[arg]
            else:
                raise Exception, arg + " not supplied in config, and no default value"
            args.append(yaml.dump(value).split('\n...')[0].rstrip()) # dump to yaml and trim end of doc markers
        return args

    def getcmd(self, config):
        """put together command invocation to run this step"""
        return [self.stepfile.path, self.verb] + self._getargs(config)


class Stepfile(object):
    def __init__(self, slinky, stepfile):
        self.slinky = slinky
        self.path = os.path.abspath(stepfile)
        self.module = os.path.split(stepfile)[1]
        self.steps = {}

        with open(self.path) as inf:
            firstline = inf.readline()

        if firstline.startswith('#!') and "python" in firstline:
            out = self._evil_fast_python_help_reader(self.path)
        else:
            out = self._general_reasonable_help_reader(self.path)

        try:
            yaml_for_out = yaml.safe_load(out)
            if yaml_for_out:
                for name, info in yaml.safe_load(out).items():
                    if name in self.steps:
                        raise Exception, "stepfile %s has duplicate definitions for '%s'" % (self.path, name)
                    self.steps[name] = Step(self, name, info)
            else:
                raise ValueError("There was no yaml.safe_load(out) for self.path `{}`:\n{}".format(self.path, out))
        except yaml.scanner.ScannerError as e:
            raise ValueError("{e} thrown while looking for step {sn} {ss} and out {out}".format(
                e=str(e), sn=self.path, ss=ss, out=out))

    @staticmethod
    def _evil_fast_python_help_reader(path):
        """super evil; share interpreter to save startup time (~5x faster); if anything goes wrong,
           consider replacing with the slower/safer general reader instead"""
        tempout, temperr = StringIO(), StringIO()
        directory, filename = os.path.split(path)
        oldstdout, oldstderr, oldargs, oldpath = sys.stdout, sys.stderr, sys.argv, sys.path
        sys.stdout, sys.stderr, sys.argv, sys.path = tempout, temperr, [filename, "help"], [directory] + sys.path
        try:
            try:
                mod = imp.new_module("__main__")
                mod.__loader__ = "evil_fast_python_help_reader"
                with open(path) as codefile:
                    exec codefile.read() in mod.__dict__
            finally:
                sys.stdout, sys.stderr, sys.argv, sys.path = oldstdout, oldstderr, oldargs, oldpath
        except Exception, e:
            print "=== reading stepfile %s failed ===" % path
            print "=== exception ==="
            traceback.print_exc(file=sys.stdout)
            print "=== stdout ==="
            print tempout.getvalue()
            print "=== stderr ==="
            print temperr.getvalue()
            print "=============="
            raise Exception, "reading stepfile failed: %s" % path
        return tempout.getvalue()

    @staticmethod
    def _general_reasonable_help_reader(path):
        p = subprocess.Popen([path, 'help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate("")
        if p.returncode:
            print "=== reading stepfile %s failed [ret=%d] ===" % (path, p.returncode)
            print "=== stdout ==="
            print out.rstrip()
            print "=== stderr ==="
            print err.rstrip()
            print "=============="
            raise Exception, "reading stepfile failed: %s" % path
        return out


class Slinky(object):
    def __init__(self, stepfiles, tagfile, statedir):
        self.definedtags = tagfile and (isinstance(tagfile, dict) and tagfile or Slinky.load_tags(tagfile)) or {}
        self.steps = self.read_stepfiles(stepfiles)
        self.stepfiles = set(s.stepfile for s in self.steps.values() if not s.tag)
        self.statedir = statedir

        if statedir is not None:
            self.logdir = os.path.join(statedir, "logs")
            self.loadlogdir = os.path.join(statedir, "loadlogs")
            self.rundir = os.path.join(statedir, "run")
            self.faildir = os.path.join(statedir, "fail")
            for dr in [self.statedir, self.logdir, self.loadlogdir, self.rundir, self.faildir]:
                if not os.path.exists(dr):
                    os.makedirs(dr)

    def read_stepfiles(self, stepfiles):
        """parse stepfiles into Stepfile and Step objects"""
        usedtags = defaultdict(list)
        allsteps = {}
        # add all steps keyed by their names
        for fn in stepfiles:
            stepfile = Stepfile(self, fn)
            for step in stepfile.steps.values():
                if step.name in allsteps:
                    raise Exception, ("duplicate definitions of %s in %s and %s" % 
                        (step.name, step.stepfile.path, allsteps[step.name].stepfile.path))
                else:
                    allsteps[step.name] = step
                # keep an index of tags each step lists
                for tag in step.tags:
                    if tag not in self.definedtags:
                        raise Exception, "tag '%s' used in %s is not defined" % (tag, step.stepfile.path)
                    usedtags[tag].append(step)

        # add tags as duck-typed steps
        for tag, steps in usedtags.items():
            if tag in allsteps:
                raise Exception, "tag '%s' duplicates step defined in %s" % (tag, allsteps[tag].stepfile.path)
            allsteps[tag] = Tag(tag, [step.name for step in steps])
        # add empty tags
        for tag in sorted(self.definedtags):
            if tag in allsteps and not allsteps[tag].tag:
                raise Exception, "tag '%s' duplicates step defined in %s" % (tag, allsteps[tag].stepfile.path)
            if tag not in allsteps:
                print >>sys.stderr, "WARNING: no steps declare tag %s" % tag
                allsteps[tag] = Tag(tag, [])
        # turn tests=[...] links into either real prereqs, or sever them from the tree
        for step in allsteps.values():
            if step.tests: # this step is a test of other steps
                tests_something = False
                for tested in step.tests:
                    if tested in allsteps:
                        # if items exist for this test, promote them to a real prereq
                        step.prereqs.append(tested)
                        tests_something = True
                if not tests_something:
                    # tests which don't test anything silently drop out of both steps and tags
                    del allsteps[step.name]
                    for otherstep in allsteps.values():
                        if otherstep.tag and step.name in otherstep.prereqs:
                            if step.name in otherstep.prereqs:
                                otherstep.prereqs.remove(step.name)
        # now go back and produce opposite pointing links
        for step in allsteps.values():
            for prereq in step.prereqs:
                if prereq in allsteps:
                    allsteps[prereq].consumers.append(step.name)

        return allsteps

    def check_config(self, goals, config, check_undos=False):
        """confirm that all config options are set for given goals"""
        missing_steps = set(goals).difference(self.steps.keys())
        if missing_steps:
            msg = """"no such steps: {m}
            goals={g}
            steps={s}""".format(m=", ".join(sorted(missing_steps)),
                                g=", ".join(sorted(goals)),
                                s=", ".join(sorted(self.steps.keys())))
            raise Exception, msg
        missing = defaultdict(set)
        checked = set() # track already checked steps, to avoid re-traversal
        def cfg_check(step, parents):
            if step in parents:
                raise Exception, "circular step graph: %s" % " > ".join([x.name for x in parents[parents.index(step):] + [step]])
            if step.name in checked:
                # stop traversing; tree already checked from this point
                # note that we have to check for circular graphs first, otherwise this will ignore them
                return True
            checked.add(step.name)
            if check_undos:
                if not step.tag and step.undo:
                    if step.undo not in self.steps:
                        raise Exception, "%s declares undefined undo step %s" % (step.name, step.undo)
                    undo_step = self.steps[step.undo]
                    for arg in undo_step.args:
                        if arg not in config:
                            missing[arg].add(step.undo)
            else:
                for arg in step.args:
                    if arg not in config:
                        missing[arg].add(step.name)
        self.traverse(cfg_check, goals, once=False, consumers=check_undos)
        if missing:
            raise Exception, "missing config: " + \
                "; ".join(["%s%s needed by %s" % (arg,
                               hasattr(config, 'unresolved') and arg in config.unresolved and(" [partially specified as '%s']"%config.unresolved[arg]) or "",
                               ", ".join(sorted(users))) for (arg, users) in missing.items()])

    def fake(self, goals, only=False, but=False):
        """by default, fake a goal and all its prereqs; alternatively, fake only a goal or only its prereqs"""
        def fakeone(step, path):
            if not step.isdone():
                print "=== faking %s" % step.name
                step.fakedone()
            else:
                print "=== %s already done" % step.name
                return True
        self._traverse_considering_only(fakeone, goals, only, but)

    def clear(self, goals, only=False, but=False):
        """by default, clear a goal and all its prereqs; alternatively, clear only a goal or only its prereqs"""
        # TODO: this default behavior is probably backwards; probably should clear a goal and all its consumers, not prereqs
        #       but putting that off as something to do when implementing "undo"
        def clearone(step, path):
            if not step.tag and step.isdone():
                print "=== clearing %s" % step.name
                step.cleardone()
            else:
                if not step.tag:
                    print "=== %s already clear" % step.name
                    return True
        self._traverse_considering_only(clearone, goals, only, but, consumers=True)

    def go(self, goals, config, spew=False, only=False, but=False, fail_later=False, recon=False):
        """choreograph execution of goals"""
        if not recon and 'blackbox' in config:
            if not os.path.exists(config['blackbox']):
                os.makedirs(config['blackbox'])
            with open(os.path.join(config['blackbox'],
                                   datetime.datetime.now().isoformat('T') + ".json"),
                      'w') as f:
                print >>f, json.dumps({"config": config, "args": sys.argv}, indent=True)
        self.check_config(goals, config)

        def prerun(step, path):
            if step.isdone():
                return True # do not descend into parents which are already done

        def recon_step(step, path):
            if step.tag:
                print "## tag %s completed" % step.name
            else:
                cmd = " ".join("'%s'" % x.replace("'", "\\'") for x in step.getcmd(config))
                if step.isdone():
                    print "## %s already done" % step.name
                    print "# " + cmd
                else:
                    print cmd

        def go_step(step, path):
            if step.tag:
                print "=== tag %s completed" % step.name
            elif step.isdone():
                print "=== %s already done" % step.name
            else:
                success, logfile = self.execute_step(step, path, config, spew)
                if success:
                    if logfile:
                        os.symlink(logfile, step.statusfile()) # mark as done
                else:
                    raise Exception, "aborting after failure of %s" % step.name
        load_monitor = None
        try:
            self._traverse_considering_only(prerun, goals, only, but, go_step if not recon else recon_step)
        finally:
            if load_monitor is not None: load_monitor.stop()

    def undo(self, goals, config, spew=False, only=False, but=False, fail_later=False):
        """choreograph execution of goals, substituting undo steps for real steps, *in reverse*"""
        self.check_config(goals, config, True) # TODO: check *undo* configs
        cant_undo = []
        def prerun(step, path):
            if not step.isdone():
                return True # do not undo parents which are not done, and don't look past them
            if not step.undo:
                cant_undo.append(step.name)
        def undo_step(step, path):
            if cant_undo:
                raise Exception, "steps cannot be undone: %s" % ", ".join(cant_undo)
            if step.tag:
                print "=== tag %s undone" % step.name
            elif not step.isdone():
                print "=== %s not done, nothing to undo" % step.name
            elif step.undo:
                success, logfile = self.execute_step(self.steps[step.undo], path, config, spew)
                if success:
                    os.unlink(step.statusfile())
                else:
                    raise Exception, "aborting after failure to undo %s" % step.name
            else:
                raise Exception, "no undo defined for %s" % step.name
        self._traverse_considering_only(prerun, goals, only, but, undo_step, consumers=True)

    @staticmethod
    def execute_step(step, path, config, spew):
        """orchestrate executing a step given the config; returns true/false for success, and logfile (if one needed)"""
        print ">>> %s" % step.name
        print "   - satisfying: %s" % (path and " < ".join([s.name for s in path[::-1]]) or "[user request]")

        # figure out where our output goes
        logfile = step.logfile()
        if os.path.exists(logfile):
            raise Exception, "cowardly refusing to replace log that already exists: " + logfile
        if not spew:
            # if just logging, open the file directly
            output, pipeproc = open(logfile, 'w'), None
        else:
            # if also printing to stdout, open a pipe to a tee process that will handle writing to disk
            output, pipeproc = pipetee(logfile)

        # do the invocation
        start = time.time()
        cmd = step.getcmd(config)
        print "   - executing:", " ".join(cmd)
        os.symlink(logfile, step.runfile())

        timeout_cmd = "timeout"
        if sys.platform == "darwin":
            timeout_cmd = "gtimeout"

        try:
            # to keep output more sane, flush stderr and stdout before starting a subprocess
            sys.stdout.flush()
            sys.stderr.flush()
            if not os.access(cmd[0], os.X_OK):
                raise Exception(cmd[0] + " is not executable")
            if not os.access(which(timeout_cmd), os.X_OK):
                raise Exception("{} is not executable/available".format(timeout_cmd))
            timeout = config.get("%s_timeout" % step.name, config["default_step_timeout"])

            retcode = subprocess.Popen([timeout_cmd, timeout] + cmd,
                                       stderr=subprocess.STDOUT,
                                       stdout=output,
                                       stdin=open("/dev/null")).wait()
        finally:
            os.unlink(step.runfile())
            output.close() # close the log, or tell tee that no more bytes are coming
            if pipeproc:
                pipeproc.wait() # if we had a tee, wait for all the bytes it has to get through to our stdout

        # handle success or failure
        if retcode:  # failed!
            try:
                if os.path.exists(step.failfile()):
                    print "{} already exists: will remove".format(step.failfile())
                    os.unlink(step.failfile())
                    assert not os.path.exists(step.failfile())
                os.symlink(logfile, step.failfile())
            except OSError, e:
                raise type(e), type(e)(e.message + str(e) +
                                       " happens for failfile {}".format(step.failfile())), sys.exc_info()[2]

            duration = time.time() - start
            print "<<< {} returned failure[{}], after {}s. {}".format(
                step.name,
                retcode,
                int(round(duration, 0)),
                "" if int(round(duration / 60.0, 0)) == 0 else "({}m)".format(int(round(duration / 60.0, 0))))
            if retcode == 124:
                print "=== Step timed out! ===\nMaximum step runtime : '%(timeout)s' " % {
                    'timeout': config.get('%s_timeout' % step.name, config['default_step_timeout'])}
            else:
                print "=== tail of output (see full output in %s) ===" % os.path.abspath(logfile)
                with open(logfile) as logf:
                    loglines = logf.read().splitlines()
                    print "\n".join(loglines[-30:])
            print "======================"

            raise Exception, "goal %s failed (see full output in %s)" % (step.name, os.path.abspath(logfile))
            # return False, logfile # if we ever decide in the future not to throw here, do this instead
        else: # succeeded!
            finish = time.time()
            duration = finish - start
            print "<<< finished {}, took {}".format(
                step.name,
                eutils.duration(duration))
            print os.path.abspath(logfile) # print full log name for quick access
            if os.path.exists(step.failfile()):
                os.unlink(step.failfile())

            # saving building times to a file for plotting
            if 'qc_plot_file' in config: # may not be in config, e.g. for slinky unit tests
                plot_file = config['qc_plot_file']
                if not os.path.exists(os.path.split(plot_file)[0]):
                    os.makedirs(os.path.split(plot_file)[0])
                if os.path.exists(plot_file):
                    temp_line_data = []
                    with open(plot_file, 'r') as f:
                        temp_line_data = open(plot_file).read().splitlines()
                    with open(plot_file, 'w') as f:
                        f.write("%s,%s\n%s,%d\n" % (temp_line_data[0], step.name, temp_line_data[1], int(duration)))
                else:
                    with open(plot_file, 'w') as f:
                        f.write("%s\n%d\n" % (step.name, int(duration)))

            return True, logfile

    def traverse(self, fn, goals, parent_path=[], once=False, seen=None, postfn=None, consumers=False):
        """traverse the step tree, calling fn(step, [parent_path, from goal to immediate parent])
           if it returns true, stop, else descend into parents
           if once is True, will skip steps as they are encountered additional times
           parent_path sets the path prefix, seen sets the already seen steps
           if postfn(step, path) is supplied, it is called after visiting all prereqs
              (which may be immediately, if fn returns True; note the return value for postfn is ignored)
           if consumers=True, will traverse in the direction of consumers instead of towards prereqs"""
        if once and seen is None:
            seen = set()
        for goal in goals:
            if goal not in self.steps:
                if parent_path:
                    raise Exception, "undefined step %s named as prerequisite of %s" % (goal, parent_path[-1].name)
                else:
                    raise Exception, "undefined step %s" % goal
            if not once or goal not in seen:
                if seen is not None:
                    seen.add(goal)
                if not fn(self.steps[goal], parent_path): # preorder traversal fn with short circuit
                    if consumers:
                        links = self.steps[goal].consumers
                    else:
                        links = self.steps[goal].prereqs
                    self.traverse(fn, links, parent_path + [self.steps[goal]], once, seen, postfn, consumers)
                if postfn:
                    postfn(self.steps[goal], parent_path) # postorder traversal fn

    def _traverse_considering_only(self, fn, goals, only=False, but=False, postfn=None, consumers=False):
        """visit steps, honoring 'only' and 'but' options"""
        assert not (only and but)
        def wrap_visit(fn):
            if not fn: return None
            def wrapper(step, path):
                if not step.tag:
                    if step.name in goals:
                        if not but:
                            return fn(step, path)
                    else:
                        if not only:
                            return fn(step, path)
                else:
                    if only and step.name in goals:
                        print "WARNING: cannot affect a tag with --only (%s)" % step.name
            return wrapper
        self.traverse(wrap_visit(fn), goals, once=True, postfn=wrap_visit(postfn), consumers=consumers)

    def generate_dot_graph(self, goals, outf):
        subgraphs = defaultdict(list)
        prereqs_edges = set()
        tags_edges = set()
        step_nodes = set()
        tag_nodes = set()
        seen = set()

        def visit(step, path):
            # always have to look at link to this step, because we may not have reached it from this path before
            if path:
                parent = path[-1]
                if parent.tag:
                    tags_edges.add((step.name, parent.name))
                else:
                    prereqs_edges.add((step.name, parent.name))
            # but we only need to continue if this is the first time we've seen the step (otherwise we've seen everything beneath)
            if step in seen:
                return True
            if step.tag:
                tag_nodes.add(step.name)
            else:
                step_nodes.add(step.name)
                #subgraphs[step.stepfile.module].append(step.name)
            seen.add(step)
        self.traverse(visit, goals)

        print >>outf, "digraph depends {"
        for node in step_nodes:
            print >>outf, '  %s [label="%s"];' % (node.replace('.', '_'), 'data' not in node and '.' in node and node.split('.', 1)[1] or node.replace(".", "\\n").replace("data_", ""))
        for node in tag_nodes:
            print >>outf, '  %s [label="%s", shape=diamond];' % (node.replace('.', '_'), '.' in node and node.split('.', 1)[1] or node)
        for f, t in prereqs_edges:
            print >>outf, "  %s -> %s;" % (f.replace('.', '_'), t.replace('.', '_'))
        for f, t in tags_edges:
            print >>outf, "  %s -> %s [style=dotted];" % (f.replace('.', '_'), t.replace('.', '_'))
        for subgraph, contents in subgraphs.items():
            if subgraph.startswith('steps_'):
                subgraph = subgraph[6:]
            print >>outf, '  subgraph cluster_%s {  %s; label="%s"; }' % \
                (subgraph, "; ".join([x.replace('.', '_') for x in contents]), subgraph)
        print >>outf, "}"

    @staticmethod
    def load_tags(tagsfile):
        return yaml.safe_load(open(tagsfile).read()) or {}

    @staticmethod
    def valid_stepfile(fullpath):
        return os.path.isfile(fullpath) and os.access(fullpath, os.X_OK) and not fullpath.endswith('~') and not fullpath.endswith(".pyc") and not fullpath.endswith('#')

    @staticmethod
    def find_stepfiles(includes_excludes):
        """takes a list of tuples ('include'/'exclude', glob) and produces a list of files"""
        stepfiles = set()
        for action, pattern in includes_excludes:
            if action == 'include':
                expand = glob(pattern)
                if not expand:
                    raise Exception, "--include '%s' does not match any files" % pattern
                for stepfile in expand:
                    if not os.access(stepfile, os.X_OK):
                        print >>sys.stderr, "WARNING: %s is not marked executable" % stepfile
                stepfiles.update(filter(Slinky.valid_stepfile, expand))
            elif action == 'exclude':
                # apply exclusion globs only to relpath
                stepfiles = set(filter((lambda fn: not fnmatch(os.path.relpath(fn, basedir), pattern)), stepfiles))
            else:
                raise Exception, "bad action: " + action
        return sorted(stepfiles)

    def _start_load_monitor(self, interval_parameter):
        if self.statedir is None:
            return None
        load_log_name = os.path.join(self.loadlogdir, "loadlog." + log_file_time_stamp() + ".tsv")
        def get_current_step(run_dir):
            files = os.listdir(run_dir)
            return files[0] if len(files) > 0 else ""
        get_current_step_func = functools.partial(get_current_step, run_dir=self.rundir)
        monitor = LoadMonitor(load_log_name, get_current_step_func, interval_parameter)
        monitor.start()
        return monitor


class Config(dict):
    def __init__(self, *confs):
        """take one or more maps, replacing values in the leftward maps with values in the rightward maps
           formatted strings are not expanded until the final step, making it possible to replace just fragments"""
        # start with some base values
        merged = dict(user=getpass.getuser())
        try:
            client = pysvn.Client()
            revision = client.info(os.path.abspath(os.path.dirname(__file__))).get("revision").number
        except NameError, x:
            revision = "unknown"
        merged.update(revision=revision)
        for conf in confs:
            merged.update(conf)
        resolved, self.unresolved = self.resolve_conf(merged)
        self.update(resolved)  # merge all reserved items into us
    
    @staticmethod
    def resolve_conf(unresolved):
        """take a config with values using string formatting; apply string formatting until they all don't;
           returns (dict of things that could be resolved, dict of things that couldn't be resolved)"""
        f = Formatter()
        resolved = {}
        while unresolved:
            changed_something, missing_defs = False, []
            for k, v in unresolved.items():
                if isinstance(v, basestring) and v and tuple(f.parse(v))[0][1] is not None:
                    try:
                        unresolved[k] = f.vformat(v, [], resolved)
                        changed_something = True
                    except KeyError, e: # missing a definition; but could in the next pass if we changed something
                        missing_defs.append(e.args[0])
                else:
                    # non strings are, by definition, resolved; no recursion into complex data structures here
                    # items without format specifiers are also resolved
                    del unresolved[k]
                    resolved[k] = v
                    changed_something = True
            if not changed_something:
                break
        return resolved, unresolved

