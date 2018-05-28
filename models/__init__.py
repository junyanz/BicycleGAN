def create_model(opt):
    model = None
    print('Loading model %s...' % opt.model)

    if opt.model == 'bicycle_gan':
        from .bicycle_gan_model import BiCycleGANModel
        model = BiCycleGANModel()
    if opt.model == 'bicycle_gan_simple':
        from .bicycle_gan_simple_model import BiCycleGANSimpleModel
        model = BiCycleGANSimpleModel()
    elif opt.model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
