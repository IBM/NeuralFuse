from easydict import EasyDict

cfg = EasyDict()

cfg.faulty_layers = ["linear", "conv"]

cfg.batch_size = 512
cfg.test_batch_size = 100
cfg.epochs = 2
cfg.precision = 8

cfg.data_dir = (
    "../dataset"
)
cfg.model_dir = (
    "model_weights/symmetric_signed/"
)
cfg.model_dir_resnetft = (
    "model_weights/symmetric_signed/resnetft/"
)
cfg.model_dir_vggft = (
    "model_weights/symmetric_signed/vggft/"
)
cfg.save_dir = (
    "~/tmp/"
)
cfg.save_dir_curve = (
    "~/tmp_curve/"
)

cfg.channels = 3

cfg.w1 = 32  # 28 #224
cfg.h1 = 32  # 28 #224
cfg.w2 = 32  # 32 28 #224
cfg.h2 = 32  # 32 28 #224
cfg.lmd = 5e-7
cfg.learning_rate = 1
cfg.decay = 0.96
cfg.max_epoch = 1
cfg.lb = 1
cfg.device = None
cfg.seed = 0


# For EOPM
cfg.N = 100
cfg.randomRange = 30000
cfg.totalRandom = True # True: Sample perturbed models in the range cfg.randomRange
cfg.G = 'ConvL'

# For transform generalization testing:
cfg.beginSeed = 50000
cfg.endSeed = 50010

# For transform_eval
cfg.testing_mode = 'generator_base' # clean / generator_base
cfg.G_PATH = ''                               