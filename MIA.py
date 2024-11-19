import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import normal
import dataset as DATA 
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
from sklearn import metrics
        
class MLP_BLACKBOX(nn.Module):
    def __init__(self, dim_in):
        super(MLP_BLACKBOX, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

def train_mia_attack_model(args, epoch, model, attack_train_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (model_loss_ori, model_trajectory, orginal_labels, predicted_labels, predicted_status, member_status) in enumerate(attack_train_loader):

        input = torch.cat((model_trajectory, model_loss_ori.unsqueeze(1)),1) 
        input = input.to(device)
        output = model(input)
        member_status = member_status.to(device)
        loss = loss_fn(output, member_status)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(member_status.view_as(pred)).sum().item()

    train_loss /= len(attack_train_loader.dataset)
    accuracy = 100. * correct / len(attack_train_loader.dataset)
    return train_loss, accuracy/100.

def test_mia_attack_model(args, epoch, model, attack_test_loader, loss_fn, max_auc, max_acc, device):
    model.eval()
    test_loss = 0
    correct = 0
    auc_ground_truth = None
    auc_pred = None
    with torch.no_grad():
        for batch_idx, (model_loss_ori, model_trajectory, orginal_labels, predicted_labels, predicted_status, member_status) in enumerate(attack_test_loader):

            input = torch.cat((model_trajectory, model_loss_ori.unsqueeze(1)),1) 
            input = input.to(device)
            output = model(input)
            member_status = member_status.to(device)
            test_loss += loss_fn(output, member_status).item()
            pred0, pred1 = output.max(1, keepdim=True)
            correct += pred1.eq(member_status.view_as(pred1)).sum().item()
            auc_pred_current = output[:, -1]
            auc_ground_truth = member_status.cpu().numpy() if batch_idx == 0 else np.concatenate((auc_ground_truth, member_status.cpu().numpy()), axis=0)
            auc_pred = auc_pred_current.cpu().numpy() if batch_idx == 0 else np.concatenate((auc_pred, auc_pred_current.cpu().numpy()), axis=0)

    test_loss /= len(attack_test_loader.dataset)
    accuracy = 100. * correct / len(attack_test_loader.dataset)

    fpr, tpr, thresholds = metrics.roc_curve(auc_ground_truth, auc_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if auc > max_auc:
        max_auc = auc
        save_data = {
            'fpr': fpr,
            'tpr': tpr
        }
        np.save(f'./outputs/{args.data}_{args.model}_{args.model_distill}_trajectory_auc', save_data)
    if accuracy > max_acc:
        max_acc = accuracy

    return test_loss, accuracy/100., auc, max_auc, max_acc

def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int] = None, return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    if labels is not None:
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            if not return_one_hot:
                labels = np.argmax(labels, axis=1)
        elif len(labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes > 2:
            labels = np.squeeze(labels)
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        elif len(labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes == 2:
            pass
        elif len(labels.shape) == 1:
            if return_one_hot:
                if nb_classes == 2:
                    labels = np.expand_dims(labels, axis=1)
                else:
                    labels = to_categorical(labels, nb_classes)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

    return labels

def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=np.int32)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1

    return categorical

def build_trajectory_membership_dataset(args, ori_model_path, device='cpu'):

    if args.model == 'vgg':
        model_name = '{}_vgg16bn'.format(args.data)
    elif args.model == 'mobilenet':
        model_name = '{}_mobilenet'.format(args.data)
    elif args.model == 'resnet':
        model_name = '{}_resnet56'.format(args.data)
    elif args.model == 'wideresnet':
        model_name = '{}_wideresnet'.format(args.data)
    elif args.model == 'lenet':
        model_name = '{}_lenet'.format(args.data)
    elif args.model == 'rnn':
        model_name = '{}_rnn'.format(args.data)
    elif args.model == 'rl':
        model_name = '{}_rl'.format(args.data)
        
    if args.mode == 'shadow':
        cnn_model, cnn_params = normal.load_model(args, ori_model_path+'/shadow', model_name, epoch=args.epochs)
    elif args.mode == 'target':
        cnn_model, cnn_params = normal.load_model(args, ori_model_path+'/target', model_name, epoch=args.epochs)

    MODEL = cnn_model.to(device)

    dataset = utils.get_dataset(cnn_params['task'], mode=args.mode, aug=True, batch_size=384)

    if args.mode == 'target':
        print('load target_dataset ... ')
        train_loader = dataset.aug_target_train_loader
        test_loader = dataset.aug_target_test_loader

    elif args.mode == 'shadow':
        print('load shadow_dataset ... ')
        train_loader = dataset.aug_shadow_train_loader
        test_loader = dataset.aug_shadow_test_loader

    model_top1 = None
    model_loss = None
    orginal_labels = None
    predicted_labels = None
    predicted_status = None
    member_status = None

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    MODEL.eval()
        
    for loader_idx, data_loader in enumerate([train_loader, test_loader]):
        top1 = DATA.AverageMeter()
        for data_idx, (data, target, ori_idx) in enumerate(data_loader):
            batch_trajectory = get_trajectory(data, target, args, ori_model_path, device)
            data, target = data.to(device), target.to(device)
            batch_logit_target = MODEL(data)

            _, batch_predict_label = batch_logit_target.max(1)
            batch_predicted_label = batch_predict_label.long().cpu().detach().numpy()
            batch_original_label = target.long().cpu().detach().numpy()
            batch_loss_target = [F.cross_entropy(batch_logit_target_i.unsqueeze(0), target_i.unsqueeze(0)) for (batch_logit_target_i, target_i) in zip(batch_logit_target, target)]
            batch_loss_target = np.array([batch_loss_target_i.cpu().detach().numpy() for batch_loss_target_i in batch_loss_target])
            batch_predicted_status = (torch.argmax(batch_logit_target, dim=1) == target).float().cpu().detach().numpy()
            batch_predicted_status = np.expand_dims(batch_predicted_status, axis=1)
            member = np.repeat(np.array(int(1 - loader_idx)), batch_trajectory.shape[0], 0)
            batch_loss_ori = batch_loss_target

            model_loss_ori               =  batch_loss_ori                  if loader_idx == 0 and data_idx == 0     else np.concatenate((model_loss_ori, batch_loss_ori), axis=0)
            model_trajectory             =  batch_trajectory                if loader_idx == 0 and data_idx == 0     else np.concatenate((model_trajectory, batch_trajectory), axis=0)
            original_labels              =  batch_original_label            if loader_idx == 0 and data_idx == 0     else np.concatenate((original_labels, batch_original_label), axis=0)
            predicted_labels             =  batch_predicted_label           if loader_idx == 0 and data_idx == 0     else np.concatenate((predicted_labels, batch_predicted_label), axis=0)
            predicted_status             =  batch_predicted_status          if loader_idx == 0 and data_idx == 0     else np.concatenate((predicted_status, batch_predicted_status), axis=0)
            member_status                =  member                          if loader_idx == 0 and data_idx == 0     else np.concatenate((member_status, member), axis=0)
            
    print(f'------------Loading trajectory {args.mode} dataset successfully!---------')
    data = {
        'model_loss_ori':model_loss_ori, 
        'model_trajectory':model_trajectory,
        'original_labels':original_labels,
        'predicted_labels':predicted_labels,
        'predicted_status':predicted_status,   
        'member_status':member_status,
        'nb_classes':dataset.num_classes
        }

    dataset_type = 'trajectory_train_data' if args.mode == 'shadow' else 'trajectory_test_data'
    utils.create_path(ori_model_path + f'/{args.mode}/{model_name}')
    np.save(ori_model_path + f'/{args.mode}/{model_name}/{dataset_type}', data)

def trajectory_black_box_membership_inference_attack(args, models_path, device='cpu'):

    if args.model == 'vgg':
        model_name = '{}_vgg16bn'.format(args.data)
    elif args.model == 'mobilenet':
        model_name = '{}_mobilenet'.format(args.data)
    elif args.model == 'resnet':
        model_name = '{}_resnet56'.format(args.data)
    elif args.model == 'wideresnet':
        model_name = '{}_wideresnet'.format(args.data)
    elif args.model == 'lenet':
        model_name = '{}_lenet'.format(args.data)
    elif args.model == 'rnn':
        model_name = '{}_rnn'.format(args.data)
    elif args.model == 'rl':
        model_name = '{}_rl'.format(args.data)
        
    if args.model_distill == 'vgg':
        model_distill_name = '{}_vgg16bn'.format(args.data)
    elif args.model_distill == 'mobilenet':
        model_distill_name = '{}_mobilenet'.format(args.data)
    elif args.model_distill == 'resnet':
        model_distill_name = '{}_resnet56'.format(args.data)
    elif args.model_distill == 'wideresnet':
        model_disltill_name = '{}_wideresnet'.format(args.data)
    elif args.model_distill == 'lenet':
        model_disltill_name = '{}_lenet'.format(args.data)
    elif args.model_distill == 'rnn':
        model_disltill_name = '{}_rnn'.format(args.data)
    elif args.model_distill == 'rl':
        model_disltill_name = '{}_rl'.format(args.data)
        
    cnn = model_name

    print(f'------------------model: {model_name}-------------------')

    orgin_model_name = model_name

    save_path = models_path + '/attack/' + model_name

    utils.create_path(save_path)

    best_prec1 = 0.0
    best_auc = 0.0
    AttackModelTrainSet = np.load(models_path + f'/shadow/{model_name}/trajectory_train_data.npy', allow_pickle=True).item()
    AttackModelTestSet = np.load(models_path + f'/target/{model_name}/trajectory_test_data.npy', allow_pickle=True).item()

    train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTrainSet['model_loss_ori'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTrainSet['model_trajectory'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['original_labels'], nb_classes=AttackModelTrainSet['nb_classes'], return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_labels'], nb_classes=AttackModelTrainSet['nb_classes'], return_one_hot=True))).type(torch.long),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTrainSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTrainSet['member_status'])).type(torch.long),)

    test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(AttackModelTestSet['model_loss_ori'], dtype='f')),
            torch.from_numpy(np.array(AttackModelTestSet['model_trajectory'], dtype='f')),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['original_labels'], nb_classes=AttackModelTestSet['nb_classes'], return_one_hot=True))).type(torch.float),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_labels'], nb_classes=AttackModelTestSet['nb_classes'], return_one_hot=True))).type(torch.long),
            torch.from_numpy(np.array(check_and_transform_label_format(AttackModelTestSet['predicted_status'], nb_classes=2, return_one_hot=True)[:,:2])).type(torch.long),
            torch.from_numpy(np.array(AttackModelTestSet['member_status'])).type(torch.long),)

    attack_train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    attack_test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)
    
    print(f'-------------------"Loss Trajectory"------------------')
    attack_model = MLP_BLACKBOX(dim_in = args.epochs_distill + 1)
    attack_optimizer = torch.optim.SGD(attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001) 
    attack_model = attack_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    max_auc = 0
    max_acc = 0

    for epoch in range(100):
        train_loss, train_prec1 = train_mia_attack_model(args, epoch, attack_model, attack_train_loader, attack_optimizer, loss_fn, device)
        val_loss, val_prec1, val_auc, max_auc, max_acc = test_mia_attack_model(args, epoch, attack_model, attack_test_loader, loss_fn, max_auc, max_acc, device)
        is_best_prec1 = val_prec1 > best_prec1
        is_best_auc = val_auc > best_auc
        if is_best_prec1:
            best_prec1 = val_prec1
        if is_best_auc:
            best_auc = val_auc
        if epoch % 10 == 0:
            print(('epoch:{} \t train_loss:{:.4f} \t test_loss:{:.4f} \t train_prec1:{:.4f} \t test_prec1:{:.4f} \t val_prec1:{:.4f} \t val_auc:{:.4f}')
                    .format(epoch, train_loss, val_loss,
                            train_prec1, val_prec1, val_prec1, val_auc))
    print('Max AUC:  ', max_auc)
    print('Max ACC:  ', max_acc/100)
    torch.save(attack_model.state_dict(), save_path + '/' + 'trajectory' + '.pkl')

    data_auc = np.load(f'./outputs/{args.data}_{args.model}_{args.model_distill}_trajectory_auc.npy', allow_pickle=True).item()
    for i in range(len(data_auc['fpr'])):
        if data_auc['fpr'][i] > 0.001:
            print('TPR at 0.1% FPR:  {:.1%}'.format(data_auc['tpr'][i-1]))
            break

def get_trajectory(data, target, args, model_path, device='cpu'):

    if args.model_distill == 'vgg':
        model_name = '{}_vgg16bn'.format(args.data)
    elif args.model_distill == 'mobilenet':
        model_name = '{}_mobilenet'.format(args.data)
    elif args.model_distill == 'resnet':
        model_name = '{}_resnet56'.format(args.data)
    elif args.model_distill == 'wideresnet':
        model_name = '{}_wideresnet'.format(args.data)
    elif args.model_distill == 'lenet':
        model_name = '{}_lenet'.format(args.data)
    elif args.model_distill == 'rnn':
        model_name = '{}_rnn'.format(args.data)
    elif args.model_distill == 'rl':
        model_name = '{}_rl'.format(args.data)
        
    trajectory = None
    predicted_label = np.array([-1]).repeat(data.shape[0],0).reshape(data.shape[0],1)

    for s in range(1):
        trajectory_current = None
        model_path_current = 'networks/{}'.format(s)
        for i in range(1, args.epochs_distill+1):
            if args.mode == 'shadow':
                cnn_model_target, cnn_params_target = normal.load_model(args, model_path_current+'/distill_shadow', model_name, epoch=i)
            elif args.mode == 'target':
                cnn_model_target, cnn_params_target = normal.load_model(args, model_path_current+'/distill_target', model_name, epoch=i)
            MODEL_target = cnn_model_target.to(device)
            data = data.to(device)
            target = target.to(device)
            logit_target = MODEL_target(data)
            loss = [F.cross_entropy(logit_target_i.unsqueeze(0), target_i.unsqueeze(0)) for (logit_target_i, target_i) in zip(logit_target, target)]
            loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
            trajectory_current = loss if i == 1 else np.concatenate((trajectory_current, loss), 1)
        trajectory = trajectory_current if s == 0 else trajectory + trajectory_current

    return trajectory






        