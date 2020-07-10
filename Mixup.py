
beta_distribution = torch.distributions.beta.Beta(alpha, alpha) # mixup 的随机数
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    # Mixup images.
    lambda_ = beta_distribution.sample([]).item()
    index = torch.randperm(images.size(0)).cuda()  # randperm功能是随机打乱一个数字序列，例如torch.randperm(4) 返回 tensor([ 2,  1,  0,  3])
    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]

    # Mixup loss.    
    scores = model(mixed_images)
    loss = (lambda_ * loss_function(scores, labels) 
            + (1 - lambda_) * loss_function(scores, labels[index]))

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

