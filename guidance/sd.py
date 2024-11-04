from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device)  # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length',
                                max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)

        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt,
                               encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * \
            (noise_pred_pos - noise_pred_uncond)

        return noise_pred

    def get_sds_loss(
        self,
        latents,
        text_embeddings,
        guidance_scale=100,
        grad_scale=1,
    ):

        # TODO: Implement the loss function for SDS
        t = torch.randint(self.min_step, self.max_step,
                          (latents.shape[0],), device=latents.device)
        noise = torch.randn_like(latents)
        latents_noisy = torch.sqrt(
            self.alphas[t]) * latents + torch.sqrt(1 - self.alphas[t]) * noise

        noise_pred = self.get_noise_preds(
            latents_noisy, t, text_embeddings, guidance_scale)
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        targets = (latents - grad).detach()
        loss = 0.5 * torch.nn.functional.mse_loss(
            latents.float(), targets, reduction='sum') / latents.shape[0]
        return loss

    def get_pds_loss(
        self, src_latents, tgt_latents,
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5,
        grad_scale=1,
    ):

        # TODO: Implement the loss function for PDS
        t = torch.randint(self.min_step, self.max_step,
                          (src_latents.shape[0],), device=src_latents.device)
        t_prev = t - 1
        noise_t = torch.randn_like(src_latents)
        noise_t_1 = torch.randn_like(src_latents)

        zts = {}
        for latents, text_embedding, name in zip(
            [tgt_latents, src_latents],
            [tgt_text_embedding, src_text_embedding],
            ["tgt", "src"]
        ):
            latents_noisy = torch.sqrt(
                self.alphas[t]) * latents + torch.sqrt(1 - self.alphas[t]) * noise_t
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            text_embeddings = torch.cat(
                [text_embedding, torch.zeros_like(text_embedding)], dim=0)
            noise_pred = self.unet(latent_model_input, tt,
                                   encoder_hidden_states=text_embeddings).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

            beta_t = self.scheduler.betas[t].to(self.device)
            alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(
                self.device)
            alpha_t = self.scheduler.alphas[t].to(self.device)
            alpha_bar_t = self.scheduler.alphas_cumprod[t].to(self.device)

            pred_x0 = (latents_noisy - torch.sqrt(1 - alpha_bar_t)
                       * noise_pred) / torch.sqrt(alpha_bar_t)
            c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
            c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / \
                (1 - alpha_bar_t)
            mu = c0 * pred_x0 + c1 * latents_noisy

            zt = (torch.sqrt(alpha_bar_t_prev) * latents - mu) / \
                torch.sqrt(1 - alpha_bar_t)
            zts[name] = zt

        grad = grad_scale * torch.nan_to_num(zts["tgt"] - zts["src"])
        target = (tgt_latents - grad).detach()

        loss = 0.5 * \
            torch.nn.functional.mse_loss(
                tgt_latents.float(), target, reduction='mean')
        return loss

    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
