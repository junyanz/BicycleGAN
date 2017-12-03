def create_model(opt):
    model = None
    print('Loading model %s...' % opt.model)

    if opt.model == 'bicycle_gan':
        from .bicycle_gan_model import BiCycleGANModel
        model = BiCycleGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
