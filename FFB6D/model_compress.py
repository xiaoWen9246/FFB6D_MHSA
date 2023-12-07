
# import torch
# from common import Config, ConfigRandLA
# from models.ffb6d import FFB6D
# from nni.compression.pytorch.pruning import L1NormPruner
# def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
#     filename = "{}.pth.tar".format(filename)

#     if os.path.isfile(filename):
#         print("==> Loading from checkpoint '{}'".format(filename))
#         ck = torch.load(filename,map_location='cpu')
#         epoch = ck.get("epoch", 0)
#         it = ck.get("it", 0.0)
#         best_prec = ck.get("best_prec", None)
#         if model is not None and ck["model_state"] is not None:
#             ck_st = ck['model_state']
#             if 'module' in list(ck_st.keys())[0]:
#                 tmp_ck_st = {}
#                 for k, v in ck_st.items():
#                     tmp_ck_st[k.replace("module.", "")] = v
#                 ck_st = tmp_ck_st
#             model.load_state_dict(ck_st)
#         if optimizer is not None and ck["optimizer_state"] is not None:
#             optimizer.load_state_dict(ck["optimizer_state"])
#         if ck.get("amp", None) is not None:
#             amp.load_state_dict(ck["amp"])
#         print("==> Done")
#         return it, epoch, best_prec
#     else:
#         print("==> ck '{}' not found".format(filename))
#         return None
# def main():

#     rndla_cfg = ConfigRandLA
#     config = Config()
#     model = FFB6D(
#             n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
#             n_kps=config.n_keypoints
#         )
#     device = torch.device('cuda:5')
#     model.to(device)
#     print(model)

# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

def plot_loss_lr(loss, lr):
    plt.plot(loss, lr)
    plt.xlabel('lr')
    plt.ylabel('loss')

