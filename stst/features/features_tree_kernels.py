# coding: utf8

from stst.modules.features import Feature
from stst.libs.kernel import tree, tree_kernels


def get_all_tree_kernels(tree1, tree2):

    k_st = tree_kernels.KernelST(l=0.95)
    k_sst = tree_kernels.KernelSST(l=0.8)
    k_p = tree_kernels.KernelPdak(l=0.8, gamma=0.8, beta=0.7)
    k_pm = tree_kernels.KernelPdakMine(l=0.8, gamma=0.8, beta=0.7)
    k_pf = tree_kernels.KernelPdakFast(l=0.8, gamma=0.8, beta=0.7)

    features = []

    for k in [k_st, k_sst, k_p, k_pm, k_pf]:
        prolog_sa = "Tree " + tree1
        sa = tree.Tree.fromPrologString(prolog_sa)

        prolog_sb = "Tree " + tree2
        sb = tree.Tree.fromPrologString(prolog_sb)

        features.append(k.kernel(sa, sb))

    return features


class POSTreeKernelFeature(Feature):

    def extract(self, train_instance):
        sa, sb = train_instance.get_parse()
        sa = sa["sentences"][0]["parse"]
        sb = sb["sentences"][0]["parse"]
        sa = sa.replace("#", "")
        sb = sb.replace("#", "")
        features = get_all_tree_kernels(sa, sb)
        infos = []

        return features, infos