# coding: utf8
"""
ref
1. http://disi.unitn.it/moschitti/Tree-Kernel.htm
2. http://disi.unitn.it/moschitti/Teaching-slides/slides-AINLP-2016/SVMs-Kernel-Methods.pdf
3. code: http://joedsm.altervista.org/pythontreekernels.htm
4. wiki: https://en.wikipedia.org/wiki/Tree_kernel
"""

import tree
import math
from copy import deepcopy

class Kernel():
    #Common routines for kernel functions
    def kernel(self,a,b):
        #compute the tree kernel on the trees a and b
        if not isinstance(a, tree.Tree):
            print "ERROR: first parameter has to be a Tree Object"
            return ""
        if not isinstance(b, tree.Tree):
            print "ERROR: second parameter has to be a Tree Object"
            return ""
        self.preProcess(a)
        self.preProcess(b)
        return self.evaluate(a,b)
    
    def preProcess(self,a):
        #Create any data structure useful for computing the kernel
        #To be instantiated in subclasses
        print "ERROR: prepProcess() must be executed in subclasses"
        pass

    def evaluate(self,a,b):
        #To be instantiated in subclasses
        print "ERROR: evaluated() must be executed in subclasses"
        pass

    def printKernelMatrix(self,dataset):
        if not isinstance(dataset, tree.Dataset):
            print "ERROR: the first Parameter must be a Dataset object"
            return
        ne = len(dataset)
        for i in range(ne):
            for j in range(i,ne):
                print "%d %d %.2f" % (i, j, self.kernel(dataset.getExample(i), dataset.getExample(j)))

class KernelST(Kernel):
    def __init__(self,l,savememory=1,hashsep="#"):
        self.l = float(l)
        self.hashsep = hashsep
        self.savememory = savememory

    def preProcess(self,a):
        if hasattr(a,'kernelstrepr'): #already preprocessed
            return 
        if not hasattr(a.root, 'stsize'):
            a.root.setSubtreeSize()
        a.root.setHashSubtreeIdentifier(self.hashsep)
        a.kernelstrepr = tree.SubtreeIDSubtreeSizeList(a.root)
        a.kernelstrepr.sort()
        if self.savememory==1:
            a.deleteRootTreeNode()
        
    def evaluate(self,a,b):
        ha, hb = (a.kernelstrepr, b.kernelstrepr)
        #Assumes ha and hb are ordered list of pairs (subtreeid, subtreesize) 
        #a.kernelreprst,b.kernelreprst are checked or created in preProcess()
        i,j,k,toti,totj = (0,0,0,len(ha), len(hb))
        while i < toti and j < totj:
            if ha.getSubtreeID(i) == hb.getSubtreeID(j):
                ci,cj=(i,j)
                while i < toti and ha.getSubtreeID(i)==ha.getSubtreeID(ci):
                    i += 1
                while j < totj and hb.getSubtreeID(j)==hb.getSubtreeID(cj):
                    j += 1
                k += (i-ci)*(j-cj)*(self.l**ha.getSubtreeSize(ci))
            elif ha.getSubtreeID(i) < hb.getSubtreeID(j):
                i += 1
            else:
                j += 1
        return k


class KernelSST(Kernel):

    def __init__(self,l,hashsep="#"):
        self.l = float(l)
        self.hashsep = hashsep
        self.cache = Cache()
    
    def preProcess(self,a):
        if hasattr(a,'kernelsstrepr'): #already preprocessed
            return 
        a.root.setHashSubtreeIdentifier(self.hashsep)
        a.kernelsstrepr = tree.ProdSubtreeList(a.root)
        a.kernelsstrepr.sort()

    def CSST(self,c,d):
        if c.getSubtreeID() < d.getSubtreeID():
            tmpkey = str(c.getSubtreeID()) + "#" + str(d.getSubtreeID())
        else:
            tmpkey = str(d.getSubtreeID()) + "#" + str(c.getSubtreeID()) 
        if self.cache.exists(tmpkey):
            return float(self.cache.read(tmpkey))
        else:
            prod = self.l
            nc = c.getOutdegree()
            if nc==d.getOutdegree():
                for ci in range(nc):
                    if c.getChild(ci).getProduction() == d.getChild(ci).getProduction():
                        prod *= (1 + self.CSST(c.getChild(ci),d.getChild(ci)))
                    else:
                        cid, did = (c.getChild(ci).getSubtreeID(),d.getChild(ci).getSubtreeID())
                        if cid < did:
                            self.cache.insert(str(cid) + str(did), 0)
                        else:
                            self.cache.insert(str(did) + str(cid), 0)
            self.cache.insert(tmpkey, prod)
        return float(prod)

    def evaluate(self,a,b):
        pa,pb=(a.kernelsstrepr, b.kernelsstrepr)
        self.cache.removeAll()
        i,j,k,toti,totj = (0,0,0,len(pa),len(pb))
        while i < toti and j < totj:
            if pa.getProduction(i) == pb.getProduction(j):
                ci,cj=(i,j)
                while i < toti and pa.getProduction(i)==pa.getProduction(ci):
                    j = cj
                    while j < totj and pb.getProduction(j)==pb.getProduction(cj):
                        k += self.CSST(pa.getTree(i),pb.getTree(j))
                        j += 1
                    i += 1
            elif len(pa.getProduction(i))<len(pb.getProduction(j)) or (len(pa.getProduction(i))==len(pb.getProduction(j)) and pa.getProduction(i) < pb.getProduction(j)):
                i += 1
            else:
                j += 1
        return k

class KernelPT(Kernel):
    def __init__(self,l,m,hashsep="#"):
        self.l = float(l)
        self.m = float(m)
        self.hashsep = hashsep
        self.cache = Cache()
    
    def preProcess(self,a):
        if hasattr(a,'kernelptrepr'): #already preprocessed
            return 
        a.root.setHashSubtreeIdentifier(self.hashsep)
        a.kernelptrepr = tree.LabelSubtreeList(a.root)
        a.kernelptrepr.sort()

    def DeltaSk(self, a, b,nca, ncb):
        DPS = [[0 for i in range(ncb+1)] for j in range(nca+1)]
        DP = [[0 for i in range(ncb+1)] for j in range(nca+1)]
        kmat = [0]*(nca+1)
        for i in range(1,nca+1):
            for j in range(1,ncb+1):
                if a.getChild(i-1).getLabel() == b.getChild(j-1).getLabel():
                    DPS[i][j] = self.CPT(a.getChild(i-1),b.getChild(j-1))
                    kmat[0] += DPS[i][j]
                else:
                    DPS[i][j] = 0
        for s in range(1,min(nca,ncb)):
            for i in range(nca+1): 
                DP[i][s-1] = 0
            for j in range(ncb+1): 
                DP[s-1][j] = 0
            for i in range(s,nca+1):
                for j in range(s,ncb+1):
                    DP[i][j] = DPS[i][j] + self.l*DP[i-1][j] + self.l*DP[i][j-1] - self.l**2*DP[i-1][j-1]
                    if a.getChild(i-1).getLabel() == b.getChild(j-1).getLabel():
                        DPS[i][j] = self.CPT(a.getChild(i-1),b.getChild(j-1))*DP[i-1][j-1]
                        kmat[s] += DPS[i][j]
        return sum(kmat)
    
    def CPT(self,c,d):
        if c.getSubtreeID() < d.getSubtreeID():
            tmpkey = str(c.getSubtreeID()) + "#" + str(d.getSubtreeID())
        else:
            tmpkey = str(d.getSubtreeID()) + "#" + str(c.getSubtreeID()) 
        if self.cache.exists(tmpkey):
            return self.cache.read(tmpkey)
        else:
            if c.getOutdegree()==0 or d.getOutdegree()==0:
                prod = self.m*self.l**2
            else:
                prod = self.m*(self.l**2+self.DeltaSk(c, d,c.getOutdegree(),d.getOutdegree()))
            self.cache.insert(tmpkey, prod)
            return prod     


    def evaluate(self,a,b):
        self.cache.removeAll()
        la,lb = (a.kernelptrepr,b.kernelptrepr)
        i,j,k,toti,totj = (0,0,0,len(la),len(lb))
        while i < toti and j < totj:
            if la.getLabel(i) == lb.getLabel(j):
                ci,cj=(i,j)
                while i < toti and la.getLabel(i) == la.getLabel(ci):
                    j = cj
                    while j < totj and lb.getLabel(j) == lb.getLabel(cj):
                        k += self.CPT(la.getTree(i),lb.getTree(j))
                        j += 1
                    i += 1
            elif la.getLabel(i) <= lb.getLabel(j):
                i += 1
            else:
                j += 1
        return k

class KernelPdak(Kernel):
    def __init__(self, l, gamma, beta, hashsep="#"):
        self.l = float(l)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.hashsep = hashsep

    def preProcess(self, t):
        if hasattr(t,'kernelpdakrepr'): #already preprocessed
            return 
        if not hasattr(t.root, 'stsize'):
            t.root.setSubtreeSize()
        t.root.setHashSubtreeIdentifier(self.hashsep)
        t.computeNodesDepth()
        t.kernelpdakrepr = tree.SubtreePositionIDLabelSubtreeSizeList(t.root)

    def mergetrees_with_depth(self, tree1, tree2):
        merge = {}
        for key in tree1:
            if key in tree2:
                merge[key] = ({(tree1[key][0],tree1[key][2]):{tree1[key][1]:1}},{(tree2[key][0],tree2[key][2]):{tree2[key][1]:1}})
                del tree2[key]
            else: merge[key] = ({(tree1[key][0],tree1[key][2]):{tree1[key][1]:1}},None)
        for key in tree2:
            merge[key] = (None,{(tree2[key][0],tree2[key][2]):{tree2[key][1]:1}})
        return merge

    def visit_with_depth(self,jtree,node,depth,param,lambda_par,gamma_par):
        kvalue = 0
        if node is not None :
            child = 0
            key = str(hash(node+'#'+str(child)))

            while key in jtree :
                kvalue = kvalue + self.visit_with_depth(jtree,key,depth+1,param,lambda_par,gamma_par)
                if jtree[key][0] is not None:
                    if jtree[node][0] is None:
                        #jtree[node][0] = jtree[key][0]
                        jtree[node] = (jtree[key][0], jtree[node][1]) 
                    else:
                        for tmpkey in jtree[key][0]:
                            if tmpkey in jtree[node][0]:
                                for tmpkey2 in jtree[key][0][tmpkey]:
                                    if tmpkey2 in jtree[node][0][tmpkey]:
                                        jtree[node][0][tmpkey][tmpkey2] = jtree[node][0][tmpkey][tmpkey2] + jtree[key][0][tmpkey][tmpkey2]
                                    else: jtree[node][0][tmpkey][tmpkey2] = jtree[key][0][tmpkey][tmpkey2]    
                            else: jtree[node][0][tmpkey] = jtree[key][0][tmpkey]
                if jtree[key][1] is not None:
                    if jtree[node][1] is None:
                        #jtree[node][1]=jtree[key][1]
                        jtree[node]=(jtree[node][0],jtree[key][1]) 
                    else:
                        for tmpkey in jtree[key][1]:
                            if tmpkey in jtree[node][1]:
                                for tmpkey2 in jtree[key][1][tmpkey]:
                                    if tmpkey2 in jtree[node][1][tmpkey]:
                                        jtree[node][1][tmpkey][tmpkey2] = jtree[node][1][tmpkey][tmpkey2] + jtree[key][1][tmpkey][tmpkey2]
                                    else: jtree[node][1][tmpkey][tmpkey2] = jtree[key][1][tmpkey][tmpkey2]
                            else: jtree[node][1][tmpkey] = jtree[key][1][tmpkey]
                child = child + 1
                key = str(hash(node+'#'+str(child)))
            # print jtree[node]
            if (jtree[node][0] is not None) and (jtree[node][1] is not None):
                for lkey in jtree[node][0]:
                    if lkey in jtree[node][1]:
                        tmpk = 0
                        for fkey1 in jtree[node][0][lkey]:
                            for fkey2 in jtree[node][1][lkey]:
                                tmpk = tmpk + lambda_par**lkey[1]*jtree[node][0][lkey][fkey1]*jtree[node][1][lkey][fkey2]*math.exp(-param*(fkey1 + fkey2))
                        kvalue = kvalue + (gamma_par**depth)*tmpk*math.exp(2*param*depth)
            return kvalue


    def evaluate(self,a,b):
        tree1 = deepcopy(a.kernelpdakrepr.sids)
        tree2 = deepcopy(b.kernelpdakrepr.sids)
        m = self.mergetrees_with_depth(tree1,tree2)
        kvalue = self.visit_with_depth(m,str(hash('0')),1,self.l, self.gamma, self.beta)
        del m, tree1, tree2
        return kvalue

class KernelPdakMine(Kernel):
    def __init__(self, l, gamma, beta, hashsep="#"):
        self.l = float(l)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.hashsep = hashsep
        self.cache = Cache()
        self.cachesize = 10000

    def preProcess(self, t):
        if hasattr(t,'kernelpdakrepr'): #already preprocessed
            return 
        if not hasattr(t.root, 'stsize'):
            t.root.setSubtreeSize()
        t.root.setHashSubtreeIdentifier(self.hashsep)
        t.computeNodesDepth()
        t.computeRoutes()
        t.kernelpdakrepr = tree.SubtreeIDSubtreeSizeRouteList(t.root)
        t.kernelpdakrepr.sort()
        #print t.kernelpdakrepr.sids

    def ntk(self, ra, da, rb, db, hra, hrb):
        if hra < hrb:
            tmpkey = str(hra) + "#" + str(hrb)
        else:
            tmpkey = str(hrb) + "#" + str(hra)
        if self.cache.exists(tmpkey):
            return float(self.cache.read(tmpkey))
        lena,lenb = len(ra), len(rb)
        c, p, minlen = 0, 0, min(lena,lenb)
        while c < minlen and ra[c] == rb[c]:
            if ra[c] == "#": p += 1
            c += 1
        #print "p = ", p, "da, db", da, db, ra, rb
        if self.gamma == 1:
            r = (p+1)*(math.e**(-self.beta*(da + db - 2*p)))
        else:
            r = (1-self.gamma**(p+1))/(1-self.gamma)*(math.e**(-self.beta*(da + db - 2*p)))
        if len(self.cache) > self.cachesize:
            self.cache.removeAll()
        self.cache.insert(tmpkey,r)
        return r
        # if self.gamma == 1:
        #     return (p+1)*(math.e**(-self.beta*(da + db - 2*p)))
        # else:
        #     return (1-self.gamma**(p+1))/(1-self.gamma)*(math.e**(-self.beta*(da + db - 2*p)))

    def evaluate(self,a,b):
        ha, hb = (a.kernelpdakrepr, b.kernelpdakrepr)
        #print ha, hb
        #Assumes ha and hb are ordered list of pairs (subtreeid, subtreesize, route) 
        #a.kernelreprst,b.kernelreprst are checked or created in preProcess()
        i,j,k,toti,totj = (0,0,0,len(ha), len(hb))
        while i < toti and j < totj:
            if ha.getLabel(i) == hb.getLabel(j):
                ci, cj = (i, j)
                while i < toti and ha.getLabel(i)==ha.getLabel(ci):
                    j = cj
                    while j < totj and hb.getLabel(j)==hb.getLabel(cj):
                        cst = self.l
                        if ha.getSubtreeID(i)==hb.getSubtreeID(j):
                            cst += self.l**ha.getSubtreeSize(i)
                            #print ha.getLabel(i), hb.getLabel(j), cst, self.ntk(ha.getRoute(i), ha.getDepth(i), hb.getRoute(j), hb.getDepth(j))
                        k += cst*self.ntk(ha.getRoute(i), ha.getDepth(i), hb.getRoute(j), hb.getDepth(j), ha.getRouteHash(i), hb.getRouteHash(j))
                        j += 1
                    i += 1
            elif ha.getLabel(i) < hb.getLabel(j):
                i += 1
            else:
                j += 1
        return k

class KernelPdakFast(KernelPdak):

    def preProcess(self, t):
        if hasattr(t,'kernelpdakrepr'): #already preprocessed
            return 
        if not hasattr(t.root, 'stsize'):
            t.root.setSubtreeSize()
        t.root.setHashSubtreeIdentifier(self.hashsep)
        t.computeNodesDepth()
        a = tree.SubtreePositionIDSubtreeIDSubtreeSizeListLabel(t.root)
        t.kernelpdakrepr = (a.sids, a.pinv)

    def mergetrees_with_depth_del_labels(self,(tree1,labels1),(tree2,labels2)):
        merge = {}
        match = 0
        for key in tree1:
            if key in tree2:
                if tree1[key][0] in labels2:
                    match = match+1
                    if tree2[key][0] in labels1:
                        merge[key] = ({(tree1[key][0],tree1[key][1]):0},{(tree2[key][0],tree2[key][1]):0})
                    else: 
                        merge[key] = ({(tree1[key][0],tree1[key][1]):0},{})
                else:
                    if tree2[key][0] in labels1:
                        merge[key] = ({},{(tree2[key][0],tree2[key][1]):0})
                        match = match+1
                    else: merge[key] = ({},{})
                del tree2[key]
            else:
                if tree1[key][0] in labels2:
                    merge[key] = ({(tree1[key][0],tree1[key][1]):0},{})
                    match = match+1
                else: merge[key] = ({},{})
        for key in tree2:
            if tree2[key][0] in labels1:
                merge[key] = ({},{(tree2[key][0],tree2[key][1]):0})
                match = match+1
            else: merge[key] = ({},{})
        return (merge,match)

    def visit_with_depth(self,jtree,node,depth,param,lambda_par,gamma_par):
        kvalue = 0
        tmpk = 0
        if node is not None :
            child = 0
            key = str(hash(node+'#'+str(child)))
            startkey = key
            max_size = [0,None]
            while key in jtree :
                kvalue = kvalue + self.visit_with_depth(jtree,key,depth+1,param,lambda_par,gamma_par)
                if (len(jtree[key][0]) + len(jtree[key][1])) > max_size[0]:
                           max_size[0] = len(jtree[key][0]) + len(jtree[key][1])
                           max_size[1] = key
                child = child + 1
                key = str(hash(node+'#'+str(child)))
            #print 'max_size',max_size[0]                                                                                                                          
            if max_size[0] > 0 :
                child = 0
                while startkey in jtree :
                  if startkey != max_size[1] :
                      if jtree[startkey][0] is not {}:
                                for tmpkey in jtree[startkey][0]:
                                    # calcolo kernel                                                                                                      
                                    if tmpkey in jtree[max_size[1]][1]:
                                        if gamma_par <> 1.0:
                                            tmpk  = tmpk + (gamma_par**(depth+1) - gamma_par)/(gamma_par - 1)*lambda_par**tmpkey[1]*jtree[startkey][0][tmpkey]*jtree[max_size[1]][1][tmpkey] 
                                        else: tmpk  = tmpk + depth*lambda_par**tmpkey[1]*jtree[startkey][0][tmpkey]*jtree[max_size[1]][1][tmpkey]
                                    # fine calcolo kernel, inizio inserimento
                      if jtree[startkey][1] is not {}:
                                for tmpkey in jtree[startkey][1]:
                                    # calcolo kernel                                                                                                      
                                    if tmpkey in jtree[max_size[1]][0]:
                                        if gamma_par <> 1.0:
                                            tmpk  = tmpk + (gamma_par**(depth+1) - gamma_par)/(gamma_par - 1)*lambda_par**tmpkey[1]*jtree[startkey][1][tmpkey]*jtree[max_size[1]][0][tmpkey]
                                        else: tmpk  = tmpk + depth*lambda_par**tmpkey[1]*jtree[startkey][1][tmpkey]*jtree[max_size[1]][0][tmpkey]
                                    # fine calcolo kernel, inizio inserimento                                                                             
                                    if tmpkey in jtree[max_size[1]][1]:
                                        jtree[max_size[1]][1][tmpkey] = jtree[max_size[1]][1][tmpkey] + jtree[startkey][1][tmpkey]
                                    else: jtree[max_size[1]][1][tmpkey] = jtree[startkey][1][tmpkey]
                    # inserisco anche hash 0 
                      for tmpkey in jtree[startkey][0]:
                        if tmpkey in jtree[max_size[1]][0]:
                            jtree[max_size[1]][0][tmpkey] = jtree[max_size[1]][0][tmpkey] + jtree[startkey][0][tmpkey]
                        else: jtree[max_size[1]][0][tmpkey] = jtree[startkey][0][tmpkey]
                  # next child 
                  child = child + 1
                  startkey = str(hash(node+'#'+str(child)))
                # fine while figli 
                if jtree[node][0] is not {}:
                    for tmpkey in jtree[node][0]:
                        # calcolo kernel                                                                                                                  
                        if tmpkey in jtree[max_size[1]][1]:
                            if gamma_par <> 1.0:
                                tmpk  = tmpk + (gamma_par**(depth+1) - gamma_par)/(gamma_par - 1)*lambda_par**tmpkey[1]*math.exp(-param*depth)*jtree[max_size[1]][1][tmpkey]
                            else: tmpk  = tmpk + depth*lambda_par**tmpkey[1]*math.exp(-param*depth)*jtree[max_size[1]][1][tmpkey]
                        # fine calcolo kernel, inizio inserimento 
                        if tmpkey in jtree[max_size[1]][0]:
                            jtree[max_size[1]][0][tmpkey] = jtree[max_size[1]][0][tmpkey] + math.exp(-param*depth)
                        else: jtree[max_size[1]][0][tmpkey] = math.exp(-param*depth)
                if jtree[node][1] is not {}:
                    for tmpkey in jtree[node][1]:
                        # calcolo kernel                                                                                      
                        if tmpkey in jtree[max_size[1]][0]:
                            if gamma_par <> 1.0:
                                tmpk  = tmpk + (gamma_par**(depth+1) - gamma_par)/(gamma_par - 1)*lambda_par**tmpkey[1]*math.exp(-param*depth)*jtree[max_size[1]][0][tmpkey]
                            else: tmpk  = tmpk + depth*lambda_par**tmpkey[1]*math.exp(-param*depth)*jtree[max_size[1]][0][tmpkey]
                        # fine calcolo kernel, inizio inserimento 
                        if tmpkey in jtree[max_size[1]][1]:
                            jtree[max_size[1]][1][tmpkey] = jtree[max_size[1]][1][tmpkey] + math.exp(-param*depth)
                        else: jtree[max_size[1]][1][tmpkey] = math.exp(-param*depth)
                jtree[node] = (jtree[max_size[1]][0],jtree[max_size[1]][1])
            else:
                for tmpkey in jtree[node][0]:
                    jtree[node][0][tmpkey] = math.exp(-param*depth)
                for tmpkey in jtree[node][1]:
                    jtree[node][1][tmpkey] = math.exp(-param*depth)
                if jtree[node][0] is not {} and jtree[node][1] is not {}:
                    for tmpkey in jtree[node][0]:
                        if tmpkey in jtree[node][1]:
                            if gamma_par <> 1.0:
                                tmpk  = tmpk + (gamma_par**(depth+1) - gamma_par)/(gamma_par - 1)*lambda_par**tmpkey[1]*jtree[node][0][tmpkey]*jtree[node][1][tmpkey]
                            else: tmpk  = tmpk + depth*lambda_par**tmpkey[1]*jtree[node][0][tmpkey]*jtree[node][1][tmpkey]
        return kvalue + tmpk*math.exp(2*param*depth)


    def evaluate(self,a,b):
        tree1 = deepcopy(a.kernelpdakrepr)
        tree2 = deepcopy(b.kernelpdakrepr)
        (m,match) = self.mergetrees_with_depth_del_labels(tree1, tree2)
        kvalue = 0
        if match > 0:
            kvalue = self.visit_with_depth(m,str(hash('0')),1,self.l, self.gamma, self.beta)
        del m, tree1, tree2
        return kvalue

####

class Cache():
    #An extremely simple cache 
    def __init__(self):
        self.cache = {} 
        self.size = 0

    def exists(self,key):
        return key in self.cache

    def existsPair(self,keya,keyb):
        if keya < keyb:
            tmpkey = str(keya) + "#" + str(keyb)
        else:
            tmpkey = str(keyb) + "#" + str(keya) 
        return tmpkey in self.cache

    def insert(self,key,value):
        self.cache[key] = value
        self.size += 1

    def insertPairIfNew(self,keya,keyb):
        if keya < keyb:
            tmpkey = str(keya) + "#" + str(keyb)
        else:
            tmpkey = str(keyb) + "#" + str(keya) 
        if not tmpkey in self.cache:
            self.insert(tmpkey)

    def remove(self,key):
        del self.cache[key]
        self.size -= 1

    def removeAll(self):
        self.cache = {}
        self.size = 0

    def read(self,key):
        return self.cache[key]

    def __len__(self):
        return self.size
