import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

from DCGAN_Model import Generator, Discriminator, initialize_weights

wandb.init(project="DCGAN-Experiment")

config = {
    "learning_rate": 1e-4,
    "batch_size": 128,
    "image_size": 64,
    "channels_img": 1,  
    "noise_dim": 100,
    "num_epochs": 50,
    "features_disc": 64,
    "features_gen": 64,
}

wandb.config.update(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize(config["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(config["channels_img"])], [0.5 for _ in range(config["channels_img"])]),
])

# Dataset: MNIST (set channels_img to 1 for grayscale images)
dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)

# Dataloader
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Models, optimizers, and loss function
gen = Generator(config["noise_dim"], config["channels_img"], config["features_gen"]).to(device)
disc = Discriminator(config["channels_img"], config["features_disc"]).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Fixed noise for generating images
fixed_noise = torch.randn(32, config["noise_dim"], 1, 1).to(device)

# Wandb logging
wandb.watch(gen, log="all")
wandb.watch(disc, log="all")

# Training loop
gen.train()
disc.train()

for epoch in range(config["num_epochs"]):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(config["batch_size"], config["noise_dim"], 1, 1).to(device)
        fake = gen(noise)

        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Log metrics and losses to wandb
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{config['num_epochs']}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")

            # Log real and fake images to wandb
            with torch.no_grad():
                fake = gen(fixed_noise)
                real_grid = torchvision.utils.make_grid(real[:32], normalize=True)
                fake_grid = torchvision.utils.make_grid(fake[:32], normalize=True)

                wandb.log({
                    "Epoch": epoch,
                    "Batch": batch_idx,
                    "Loss Disc": loss_disc.item(),
                    "Loss Gen": loss_gen.item(),
                    "Real Images": [wandb.Image(real_grid, caption="Real Images")],
                    "Fake Images": [wandb.Image(fake_grid, caption="Fake Images")],
                })
