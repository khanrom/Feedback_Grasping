from predify.modules import PCoderN
from predify.networks import PNetSeparateHP
from torch.nn import Sequential, ReLU, ConvTranspose2d

class PVGG16SeparateHP(PNetSeparateHP):
    def __init__(self, backbone, build_graph=False, random_init=False, ff_multiplier=(0.35,0.4), fb_multiplier=(0.25,0.0), er_multiplier=(0.01,0.01)):
        super().__init__(backbone, 2, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        # PCoder number 1                                                                     \
                                                                                               
        pmodule = ConvTranspose2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.pcoder1 = PCoderN(pmodule, True, self.random_init)
        self.target_layer = None
        def hook_fn(module, input, output):
            self.target_layer = output
        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.target_layer, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.features_compress[18].register_forward_hook(hook_fn)
        self.backbone.features_compress[21].register_forward_hook(fw_hook1)

        # PCoder number 2                                                                     \
                                                                                               
        pmodule = Sequential(ConvTranspose2d(256, 256, kernel_size=(10, 10), stride=(2, 2), padding=(4, 4)), ReLU(inplace=True))
        self.pcoder2 = PCoderN(pmodule, False, self.random_init)
        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=None, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.backbone.features_compress[28].register_forward_hook(fw_hook2)
