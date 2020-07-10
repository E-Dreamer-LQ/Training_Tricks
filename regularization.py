### add in 2019.6.8 L1 and L2 regularization
class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.weight_list = self.get_weight(model)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay)

        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay,L1 = False,L2= True):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        reg_loss = 0
        for name, w in weight_list:
            l1_reg = torch.norm(w,p=1)
            l2_reg = torch.norm(w,p=2)
            if L1 and L2:
                reg_loss = reg_loss + l2_reg + l1_reg
            if not L1 and L2:
                reg_loss = reg_loss + l2_reg
            if L1 and not L2:
                reg_loss = reg_loss + l1_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss


if __name__ == "__main__":
    weight_decay = 0.00005
    if weight_decay > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reg_loss = Regularization(FPN, weight_decay).to(device) #  FPN 为网络模型
    else:
        print("no regularization")

    

