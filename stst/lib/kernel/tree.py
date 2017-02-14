# coding: utf8
"""
A set of Classes to handle trees and compute kernel functions on them
"""

import random
import bisect


class TreeNode:
    # A simple class for handling tree nodes
    def __init__(self, val=None, chs=[]):
        self.val = val  # node label
        self.chs = chs  # list of children of the node

    @classmethod
    def fromPrologString(cls, s):
        # to be invoked as tree.TreeNode.fromPrologString(s)
        # where s is the string encoding the tree
        # The format is as follows:
        s = s.rstrip('\n')  # remove trailing newlines
        i, tmps, lens = (0, "", len(s))
        while i < lens and s[i] in "-+0123456789":
            tmps += s[i]
            i += 1
        if i >= lens or s[i] != " ":
            i = 0
        else:
            i += 1
        aa = []
        while (i < lens):
            tmps = ""
            while (i < lens) and s[i] not in ('(', ')', ','):
                tmps += s[i]
                i += 1
            if len(tmps) > 0:
                t = cls(tmps, [])
                if len(aa) > 0:
                    aa[-1].chs.append(t)
            if i < lens:
                if s[i] == '(':
                    aa.append(t)
                elif s[i] == ')':
                    t = aa.pop()
                elif s[i] == ',':
                    pass
            i += 1
        return t

    def tostring_prolog(self):
        # returns a string in which the subtree rooted
        # at self is represented in prolog-style
        if not self:
            return
        stri = ""
        # if hasattr(self,'chs') and
        if self.chs:
            stri += "%s(" % self.val
            for i, c in enumerate(self.chs):
                stri += c.tostring_prolog()
                if i < len(self.chs) - 1:
                    stri += ","
            stri += ")"
        else:
            stri += "%s" % self.val
        return stri

    def __str__(self):
        # return the tree in prolog format
        return self.tostring_prolog()

    def tostring_svmlight(self):
        # returns a string in which the subtree rooted
        # at self is represented in svmlight-style
        if not self:
            return
        stri = ""
        if self.chs:
            stri += "(%s " % self.val
            for i, c in enumerate(self.chs):
                stri += c.tostring_svmlight()
            stri += ")"
        else:
            stri += "(%s -)" % self.val
        return stri

    def getLabel(self):
        return self.val

    def getChild(self, i):
        return self.chs[i]

    def getChildren(self):
        return self.chs

    def getOutdegree(self):
        if not self:
            return 0
        else:
            return len(self.chs)

    def getMaxOutdegree(self):
        if not self:
            return 0
        else:
            m = self.getOutdegree()
            for c in self.chs:
                m = max(m, c.getMaxOutdegree())
        return m

    def getNodeLabelList(self):
        # returns the list of labels of all descendants of self
        if not self: return []
        p = [self.val]
        for c in self.chs:
            p.extend(c.getNodeLabelList())
        return p

    def labelList(self):
        # returns the list of labels of all descendants of self
        if not self: return []
        p = [(self.val, self)]
        for c in self.chs:
            p.extend(c.labelList())
        return p

    def getProduction(self):
        # return the label of the node concatenated with the labels of its children
        if not self: return ""
        self.production = self.val + "(" + ','.join([c.val for c in self.chs]) + ")"
        return self.production

    def productionlist(self):
        # returns the list of productions of all nodes
        # in the subtree rooted at self
        if not self: return []
        p = [(self.getProduction(), self)]
        for c in self.chs:
            p.extend(c.productionlist())
        return p

    def getSubtreeSize(self):
        # returns the number of nodes in the subtree rooted at self
        if not self:
            return 0
        n = 1
        for c in self.chs:
            n += c.getSubtreeSize()
        return n

    def setSubtreeSize(self):
        # returns the number of nodes in the subtree rooted at self
        # for each visited node A such value is stored in A.stsize
        if not self:
            self.stsize = 0
            return 0
        n = 1
        for c in self.chs:
            n += c.setSubtreeSize()
        self.stsize = n
        return n

    def setSubtreeRoutes(self, r=""):
        self.route = r
        i = 1
        for c in self.chs:
            c.setSubtreeRoutes(r + str(i) + "#")
            i += 1

    def getDepth(self):
        if not hasattr(self, 'depth'):
            print "ERROR: node depth has not been computed!"
            return ""
        return self.depth

    def setDepth(self, subtreerootdepth=0):
        # compute the depth (w.r.t self) of the descendants of self
        if not self:
            return
        self.depth = subtreerootdepth
        for c in self.chs:
            c.setDepth(subtreerootdepth + 1)
        return

    def height(self):
        # returns the length of the longest path
        # connecting self to its farthest leaf
        if not self:
            return 0
        p = 0
        for c in self.chs:
            p = max(p, c.height())
        return p + 1

    def getSubtreeID(self):
        return self.subtreeId

    def getLabelFrequencies(self):
        lab = {}
        lab[self.val] = 1
        for c in self.chs:
            l = c.getLabelFrequencies()
            for lk in l.keys():
                if not lk in lab:
                    lab[lk] = l[lk]
                else:
                    lab[lk] += l[lk]
        return lab

    def getHashSubtreeIdentifier(self, sep):
        # compute an hash value from the label of the node
        # self and the hash values of the children of self
        if not self: return
        stri = self.val
        for c in self.chs:
            stri += sep + c.getHashSubtreeIdentifier()
        return str(hash(stri))

    def setHashSubtreeIdentifier(self, sep):
        # compute an hash value from the label of the node
        # self and the hash values of the children of self
        # For each visited node A the hash value is stored
        # into A.hash
        if not self: return
        stri = self.val
        if stri.find(sep) != -1:
            print "ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)"
        for c in self.chs:
            stri += sep + c.setHashSubtreeIdentifier(sep)
        self.subtreeId = str(hash(stri))
        return self.subtreeId

    def computeSubtreeIDSubtreeSizeList(self):
        # compute a list of pairs (subtree-hash-identifiers, subtree-size)
        if not self:
            return
        p = [(self.subtreeId, self.stsize)]
        for c in self.chs:
            p.extend(c.computeSubtreeIDSubtreeSizeList())
        return p

    def computeSubtreeIDSubtreeSizeRouteList(self):
        if not self:
            return
        p = [(self.val, self.subtreeId, self.stsize, self.depth, self.route)]
        for c in self.chs:
            p.extend(c.computeSubtreeIDSubtreeSizeRouteList())
        return p

    def computeSubtreeIDSubtreeSizeRouteRouteHashList(self, h):
        if not self:
            return
        p = [(self.val, self.subtreeId, self.stsize, self.depth, self.route, h)]
        i = 1
        for c in self.chs:
            p.extend(c.computeSubtreeIDSubtreeSizeRouteRouteHashList(str(hash(h + "#" + str(i)))))
            i += 1
        return p

    def computeSubtreePositionIDLabelSubtreeSizeList(self, h):
        # compute a hash whose key is the subtree-position-identifier and the value
        # is a triplet (subtree-hash-identifiers, node depth, subtree-size)
        # A key is constructed for each node
        if not self:
            return
        p = {}
        p[h] = (self.subtreeId, self.getDepth(), self.stsize)
        i = -1
        for c in self.chs:
            i += 1
            p.update(c.computeSubtreePositionIDLabelSubtreeSizeList(str(hash(h + "#" + str(i)))))
        return p

    def computeSubtreePositionIDSubtreeIDSubtreeSizeListLabel(self, h):
        if not self:
            return
        p, pinv = ({}, {})
        p[h] = (self.subtreeId, self.stsize)
        pinv[self.subtreeId] = h
        i = -1
        for c in self.chs:
            i += 1
            (tmpp, tmppinv) = c.computeSubtreePositionIDSubtreeIDSubtreeSizeListLabel(str(hash(h + "#" + str(i))))
            p.update(tmpp)
            pinv.update(tmppinv)
        return (p, pinv)

    def computeSubtreeIDTreeNodeList(self):
        if not self:
            return
        p = [(self.subtreeId, self)]
        for c in self.chs:
            p.extend(c.computeSubtreeIDTreeNode())
        return p


class RandomTrees():
    # A class for generating random trees
    def __init__(self, p, d, outdegree, nodelabels):
        self.p = p
        self.d = d
        self.outdegree = outdegree
        self.nodelabels = nodelabels

    def __newTree(self, p):
        if random.random() > p:
            return None
        chs = []
        for i in range(self.outdegree):
            t = self.__newTree(p * self.d)
            if t: chs.append(t)
        return TreeNode(self.randomLabel(), chs)

    def newTree(self):
        t = self.__newTree(self.p)
        while not t:
            t = self.__newTree(self.p)
        return t

    def randomLabel(self):
        return random.choice(self.nodelabels)


class RandomTreesPowerLawDistribution(RandomTrees):
    # A class for generating random trees where labels are selected
    # randomly according zipf distribution (first elements
    # have much higher probability to be selected than last ones)
    def __init__(self, p, d, outdegree, numberoflabels):
        RandomTrees.__init__(self, p, d, outdegree, [])
        s = 0.99
        # self.labelfrequency = [0]*numberoflabels
        self.nodelabels = [1 / (i ** s) for i in range(1, numberoflabels + 1)]
        norm = sum(self.nodelabels)
        self.nodelabels = [x / norm for x in self.nodelabels]
        cpd = 0
        for i in range(0, numberoflabels):
            cpd += self.nodelabels[i]
            self.nodelabels[i] = cpd

    def randomLabel(self):
        r = bisect.bisect(self.nodelabels, random.random())
        # self.labelfrequency[r] += 1
        return r


class Tree:
    # A tree instance suitable for being processed by a tree kernel
    # A TreeNode retain properties of single nodes, a Tree a property
    # of a set of nodes: max/average outdegree, max depth
    def __init__(self, root, target=""):
        self.root = root
        self.target = target

    @classmethod
    def fromPrologString(cls, s):
        # to be invoked as tree.Tree.fromPrologString(s)
        # where s is the string encoding the tree
        target, i, tmps = ("", 0, "")
        while s[i] in "-+0123456789":
            tmps += s[i]
            i += 1
        if len(tmps) > 0 and s[i] == " ":  # the target is valid
            target = tmps
        return cls(TreeNode.fromPrologString(s), target)

    def deleteRootTreeNode(self):
        self.root = None

    def getMaxDepth(self):
        if not hasattr(self.root, 'maxdepth'):
            return self.root.height()
        else:
            return self.maxdepth

    def computeNodesDepth(self):
        self.root.setDepth()

    def setMaxDepth(self):
        self.maxdepth = self.root.height()

    def getMaxOutdegree(self):
        if not self.root:
            return 0  # ERROR?
        else:
            return self.root.getMaxOutdegree()

    def getLabelFrequencies(self):
        if not self.root:
            return {}
        else:
            return self.root.getLabelFrequencies()

    def __str__(self):
        if self.target:
            return str(self.target) + " " + str(self.root)
        else:
            return str(self.root)

    def printFormat(self, frmt="prolog"):
        s = ""
        if self.target:
            s = str(self.target) + " "
        if frmt == "prolog":
            s += self.root.tostring_prolog()
        elif frmt == "svmlight":
            s += "|BT| " + self.root.tostring_svmlight() + " |ET| "
        return s

    def computeSubtreeIDs(self, hashsep):
        self.root.setHashSubtreeIdentifier(hashsep)

    def computeRoutes(self, r=""):
        self.root.setSubtreeRoutes(r)


class SubtreeIDSubtreeSizeList():
    def __init__(self, root):
        self.sids = root.computeSubtreeIDSubtreeSizeList()

    def getSubtreeID(self, i):
        return self.sids[i][0]

    def getSubtreeSize(self, i):
        return self.sids[i][1]

    def sort(self):
        self.sids.sort()

    def __len__(self):
        return len(self.sids)


class ProdSubtreeList():
    def __init__(self, root):
        self.prodorderedlist = root.productionlist()

    def getProduction(self, i):
        return self.prodorderedlist[i][0]

    def getTree(self, i):
        return self.prodorderedlist[i][1]

    def sort(self):
        self.prodorderedlist.sort(cmp=lambda x, y: cmp(x[0], y[0]))
        self.prodorderedlist.sort(cmp=lambda x, y: cmp(len(x[0]), len(y[0])))

    def __len__(self):
        return len(self.prodorderedlist)

    def compareprods(x, y):
        if len(x[0]) == len(y[0]):
            return cmp(x[0], y[0])
        else:
            return cmp(len(x[0]), len(y[0]))


class LabelSubtreeList():
    def __init__(self, root):
        self.labelList = root.labelList()

    def getLabel(self, i):
        return self.labelList[i][0]

    def getTree(self, i):
        return self.labelList[i][1]

    def sort(self):
        self.labelList.sort(cmp=lambda x, y: cmp(x[0], y[0]))

    def __len__(self):
        return len(self.labelList)


class SubtreePositionIDLabelSubtreeSizeList():
    def __init__(self, root):
        self.sids = root.computeSubtreePositionIDLabelSubtreeSizeList(str(hash('0')))

    def getSubtreeID(self, i):
        return self.sids[i][0][0]

    def getLabel(self, i):
        return self.sids[i][1]

    def getSubtreeSize(self, i):
        return self.sids[i][0][2]

    def __len__(self):
        return len(self.sids)


class SubtreePositionIDSubtreeIDSubtreeSizeListLabel():
    def __init__(self, root):
        (self.sids, self.pinv) = root.computeSubtreePositionIDSubtreeIDSubtreeSizeListLabel(str(hash('0')))

    def getSubtreeID(self, i):
        return self.sids[i][0]

    def getPositionID(self, label):
        return self.pinv[label]

    def getSubtreeSize(self, i):
        return self.sids[i][1]

    def __len__(self):
        return len(self.sids)


class SubtreeIDSubtreeSizeRouteList():
    # Currently used by Tree Kernel class PdakMine
    def __init__(self, root):
        self.sids = root.computeSubtreeIDSubtreeSizeRouteRouteHashList("0")

    def getLabel(self, i):
        return self.sids[i][0]

    def getSubtreeID(self, i):
        return self.sids[i][1]

    def getSubtreeSize(self, i):
        return self.sids[i][2]

    def getDepth(self, i):
        return self.sids[i][3]

    def getRoute(self, i):
        return self.sids[i][4]

    def getRouteHash(self, i):
        return self.sids[i][5]

    def sort(self):
        self.sids.sort(cmp=lambda x, y: cmp(x[1], y[1]))
        self.sids.sort()

    def __len__(self):
        return len(self.sids)


class Dataset():
    # A class for handling a collection of Tree Objects
    def __init__(self, treeList=[]):
        self.examples = treeList

    def loadFromFilePrologFormat(self, filename):
        self.filename = filename
        self.examples = []
        f = open(filename, "r")
        for line in f:
            self.examples.append(self.loadExamplePrologFormat(line))
        f.close()

    def generateRandomDataset(self, randObj, numberofexamples):
        self.examples = []
        for i in range(numberofexamples):
            self.examples.append(Tree(randObj.newTree(), 1))

    def __len__(self):
        return len(self.examples)

    def getExample(self, i):
        return self.examples[i]

    def loadExamplePrologFormat(self, line):
        return Tree.fromPrologString(line)

    def __len__(self):
        return len(self.examples)

    def getTotalNumberOfNodes(self):
        if hasattr(self, 'totalnodes'):
            return self.totalnodes
        else:
            s = 0
            for i in range(len(self.examples)):
                s += self.examples[i].root.getSubtreeSize()
            return s

    def setTotalNumberOfNodes(self):
        self.totalnodes = self.getTotalNumberOfNodes()

    def getNodesNumberAverage(self):
        return self.getTotalNumberOfNodes() / len(self.examples)

    def getNodesNumberVariance(self):
        avg = self.getNodesNumberAverage()
        s = 0
        for i in range(len(self.examples)):
            s += (avg - len(self.examples[i])) ** 2
        return s / (len(self.examples))

    def getAverageMaxOutdegree(self):
        o = 0
        for i in range(len(self.examples)):
            o += self.examples[i].getMaxOutdegree()
        return o

    def getMaxMaxOutdegree(self):
        o = 0
        for i in range(len(self.examples)):
            o = max(o, self.examples[i].getMaxOutdegree())
        return o

    def getMaxAndAverageMaxOutdegree(self):
        o, m = (0, 0)
        for i in range(len(self.examples)):
            cm = self.examples[i].getMaxOutdegree()
            o += cm
            m = max(m, cm)
        return o, m

    def random_permutation(self, seed):
        pass

    def getLabelFrequencies(self):
        lab = {}
        for i in range(len(self.examples)):
            l = self.examples[i].getLabelFrequencies()
            for lk in l.keys():
                if lk not in lab:
                    lab[lk] = l[lk]
                else:
                    lab[lk] += l[lk]
        return lab

    def getStats(self):
        self.setTotalNumberOfNodes()
        avgo, maxo = self.getMaxAndAverageMaxOutdegree()
        s = "%f %d %d %f" % (self.getNodesNumberAverage(), self.getTotalNumberOfNodes(), maxo, avgo)
        return s

    def printToFile(self, filename):
        f = open(filename, "w")
        for i in range(len(self.examples)):
            f.write(str(self.examples[i]) + "\n")
        f.close()

    def printToFileSvmlightFormat(self, filename):
        f = open(filename, "w")
        for i in range(len(self.examples)):
            f.write(str(self.examples[i].printFormat("svmlight")) + "\n")
        f.close()
