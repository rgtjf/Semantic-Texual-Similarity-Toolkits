import tree_kernels
import tree
#
# lambda # kernel parameter

# PrologString_sa = "t(a t(b t(d nil nil) t(e nil nil)) t(c nil t(f t(g nil nil) nil))) "
# PrologString_sb = "t(a t(b t(d nil nil) t(e nil nil)) t(c nil t(f t(g nil nil) nil))) "
PrologString_sa = "Tree (ROOT\n  (S\n    (NP (DT A) (NN man))\n    (VP (VBZ is)\n      (VP (VBG carrying)\n        (NP (DT a) (NN canoe))\n        (PP (IN with)\n          (NP (DT a) (NN dog)))))\n    (. .)))"


PrologString_sb = "Tree (ROOT\n  (S\n    (NP (DT A) (NN dog))\n    (VP (VBZ is)\n      (VP (VBG carrying)\n        (NP\n          (NP (DT a) (NN man))\n          (PP (IN in)\n            (NP (DT a) (NN canoe))))))\n    (. .)))"

PrologString_sc = "Tree (ROOT\n  (S\n    (NP (DT A) (NN man))\n    (VP (VBZ is)\n      (VP (VBG carrying)\n        (NP\n          (NP (DT a) (NN dog))\n          (PP (IN in)\n            (NP (DT a) (NN canoe))))))\n    (. .)))"

PrologString_sd = "Tree (ROOT\n  (NP\n    (NP (DT A) (NN girl) (NN dancing))\n    (PP (IN on)\n      (NP (DT a) (JJ sandy) (NN beach)))\n    (. .)))"



sa = tree.Tree.fromPrologString(PrologString_sa)
sb = tree.Tree.fromPrologString(PrologString_sb)
sc = tree.Tree.fromPrologString(PrologString_sc)
sd = tree.Tree.fromPrologString(PrologString_sd)

# k = tree_kernels.KernelST(l=0.95)
# k = tree_kernels.KernelSST(l=0.5)
# k = tree_kernels.KernelPdak(l=0.5, gamma=0.5, beta=0.5)
# print(k.kernel(sb, sb))
# print(k.kernel(sc, sb))
# print(k.kernel(sa, sb))
# print(k.kernel(sd, sb))

# k = tree_kernels.KernelPdakMine(l=0.5, gamma=0.5, beta=0.5)
# print(k.kernel(sb, sb))
# print(k.kernel(sc, sb))
# print(k.kernel(sa, sb))
# print(k.kernel(sd, sb))

# k = tree_kernels.KernelPdakFast(l=0.5, gamma=0.5, beta=0.5)
# print(k.kernel(sb, sb))
# print(k.kernel(sc, sb))
# print(k.kernel(sa, sb))
# print(k.kernel(sd, sb))

# k.printKernelMatrix(dat)
