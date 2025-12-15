import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttnPool1D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Conv1d(d_model, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        w = self.score(feat).squeeze(1)
        w = torch.softmax(w, dim=-1)
        pooled = torch.einsum("bdt,bt->bd", feat, w)
        return pooled


class ApneaDetectionModel(nn.Module):
    def __init__(
        self,
        n_bands: int = 5,
        n_channels: int = 3,
        n_stages: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_tokens = n_bands * n_channels

        # cnn applied to each channel
        self.backbone = CNN1D(d_model=d_model, dropout=dropout)
        self.pool = AttnPool1D(d_model=d_model)

        # token embeddings
        self.band_emb = nn.Embedding(n_bands, d_model)
        self.chan_emb = nn.Embedding(n_channels, d_model)

        # stage embedding
        self.stage_emb = nn.Embedding(n_stages, d_model)
        self.stage_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # per-band cross-attention
        self.band_to_chan_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )

        # per-channel cross-attention
        self.chan_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )

        # layer normalization
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff0 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        # classifier mlp
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        nn.init.normal_(self.stage_token, std=0.02)

    def forward(self, eeg_bands: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        B, n_bands, n_ch, T = eeg_bands.shape
        assert n_bands == self.n_bands and n_ch == self.n_channels

        # cnn feature extraction
        x = eeg_bands.reshape(B * self.n_tokens, 1, T)

        feat = self.backbone(x)
        tok = self.pool(feat)
        tok = tok.reshape(B, self.n_tokens, -1)

        # band and channel embeddings
        band_ids = torch.arange(self.n_bands, device=tok.device).repeat_interleave(self.n_channels)
        chan_ids = torch.arange(self.n_channels, device=tok.device).repeat(self.n_bands)
        tok = tok + self.band_emb(band_ids)[None, :, :] + self.chan_emb(chan_ids)[None, :, :]

        # stage embedding
        s = self.stage_emb(stage).unsqueeze(1)
        stage_tok = self.stage_token.expand(B, -1, -1) + s

        tok_bc = tok.view(B, self.n_bands, self.n_channels, -1).permute(0, 2, 1, 3).contiguous()

        chan_summaries = []
        for c in range(self.n_channels):
            band_tokens_c = tok_bc[:, c, :, :]
            q = stage_tok
            attn_out, _ = self.band_to_chan_attn(q, band_tokens_c, band_tokens_c)
            y = self.norm0(attn_out + q)
            y = self.norm0(y + self.ff0(y))
            chan_summaries.append(y)

        chan_sum = torch.cat(chan_summaries, dim=1)

        # cross-attention
        attn_out, _ = self.chan_cross_attn(chan_sum, chan_sum, chan_sum)
        z = self.norm1(attn_out + chan_sum)
        z = self.norm2(z + self.ff1(z))

        # mean pooling
        z = z.mean(dim=1)

        # classifier output
        logits = self.head(z).squeeze(-1)
        return logits


class ApneaDetectionModelFewerBands(nn.Module):
    def __init__(
        self,
        n_bands: int = 3,
        n_channels: int = 3,
        n_stages: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_tokens = n_bands * n_channels

        # cnn applied to each channel
        self.backbone = CNN1D(d_model=d_model, dropout=dropout)
        self.pool = AttnPool1D(d_model=d_model)

        # token embeddings
        self.band_emb = nn.Embedding(n_bands, d_model)
        self.chan_emb = nn.Embedding(n_channels, d_model)

        # stage embedding
        self.stage_emb = nn.Embedding(n_stages, d_model)
        self.stage_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # per-band cross-attention
        self.band_to_chan_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )

        # per-channel cross-attention
        self.chan_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )

        # layer norms and ffns
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff0 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        # classifier mlp
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        nn.init.normal_(self.stage_token, std=0.02)

    def forward(self, eeg_bands: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        B, n_bands, n_ch, T = eeg_bands.shape
        assert n_bands == self.n_bands and n_ch == self.n_channels

        # cnn feature extraction
        x = eeg_bands.reshape(B * self.n_tokens, 1, T)

        feat = self.backbone(x)
        tok = self.pool(feat)
        tok = tok.reshape(B, self.n_tokens, -1)

        # band and channel embeddings
        band_ids = torch.arange(self.n_bands, device=tok.device).repeat_interleave(self.n_channels)
        chan_ids = torch.arange(self.n_channels, device=tok.device).repeat(self.n_bands)
        tok = tok + self.band_emb(band_ids)[None, :, :] + self.chan_emb(chan_ids)[None, :, :]

        # stage embedding
        s = self.stage_emb(stage).unsqueeze(1)
        stage_tok = self.stage_token.expand(B, -1, -1) + s

        tok_bc = tok.view(B, self.n_bands, self.n_channels, -1).permute(0, 2, 1, 3).contiguous()

        # attention
        chan_summaries = []
        for c in range(self.n_channels):
            band_tokens_c = tok_bc[:, c, :, :]
            q = stage_tok
            attn_out, _ = self.band_to_chan_attn(q, band_tokens_c, band_tokens_c)
            y = self.norm0(attn_out + q)
            y = self.norm0(y + self.ff0(y))
            chan_summaries.append(y)

        chan_sum = torch.cat(chan_summaries, dim=1)

        attn_out, _ = self.chan_cross_attn(chan_sum, chan_sum, chan_sum)
        z = self.norm1(attn_out + chan_sum)
        z = self.norm2(z + self.ff1(z))

        # mean pooling
        z = z.mean(dim=1)

        # classifier output
        logits = self.head(z).squeeze(-1)
        return logits


class ApneaDetectionModelNoAttention(nn.Module):
    def __init__(
        self,
        n_bands: int = 5,
        n_channels: int = 3,
        n_stages: int = 4,
        d_model: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_tokens = n_bands * n_channels

        # cnn applied to each channel
        self.backbone = CNN1D(d_model=d_model, dropout=dropout)
        self.pool = AttnPool1D(d_model=d_model)

        # token embeddings
        self.band_emb = nn.Embedding(n_bands, d_model)
        self.chan_emb = nn.Embedding(n_channels, d_model)

        # stage embedding
        self.stage_emb = nn.Embedding(n_stages, d_model)

        # simple mlp instead of attention
        self.fusion = nn.Sequential(
            nn.Linear(d_model * (self.n_tokens + 1), 4 * d_model),
            nn.LayerNorm(4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # classifier mlp
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, eeg_bands: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        B, n_bands, n_ch, T = eeg_bands.shape
        assert n_bands == self.n_bands and n_ch == self.n_channels

        # cnn feature extraction
        x = eeg_bands.reshape(B * self.n_tokens, 1, T)

        feat = self.backbone(x)
        tok = self.pool(feat)
        tok = tok.reshape(B, self.n_tokens, -1)

        # band and channel embeddings
        band_ids = torch.arange(self.n_bands, device=tok.device).repeat_interleave(self.n_channels)
        chan_ids = torch.arange(self.n_channels, device=tok.device).repeat(self.n_bands)
        tok = tok + self.band_emb(band_ids)[None, :, :] + self.chan_emb(chan_ids)[None, :, :]

        # stage embedding
        s = self.stage_emb(stage)

        # concatenate all tokens with stage embedding
        all_features = torch.cat([tok.view(B, -1), s], dim=1)

        # simple feedforward fusion
        z = self.fusion(all_features)

        # classifier output
        logits = self.head(z).squeeze(-1)
        return logits


class ApneaDetectionModelSingleChannel(nn.Module):
    def __init__(
        self,
        n_bands: int = 5,
        n_channels: int = 1,
        n_stages: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_tokens = n_bands * n_channels

        # cnn applied to each channel
        self.backbone = CNN1D(d_model=d_model, dropout=dropout)
        self.pool = AttnPool1D(d_model=d_model)

        # token embeddings
        self.band_emb = nn.Embedding(n_bands, d_model)

        # stage embedding
        self.stage_emb = nn.Embedding(n_stages, d_model)
        self.stage_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # only single cross-attention
        self.band_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )

        # layer norms and ffns
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff0 = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        # classifier mlp
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        nn.init.normal_(self.stage_token, std=0.02)

    def forward(self, eeg_bands: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        B, n_bands, n_ch, T = eeg_bands.shape
        assert n_bands == self.n_bands and n_ch == self.n_channels

        # cnn feature extraction
        x = eeg_bands.reshape(B * self.n_tokens, 1, T)

        feat = self.backbone(x)
        tok = self.pool(feat)
        tok = tok.reshape(B, self.n_tokens, -1)

        # band embeddings only
        band_ids = torch.arange(self.n_bands, device=tok.device)
        tok = tok + self.band_emb(band_ids)[None, :, :]

        # stage embedding
        s = self.stage_emb(stage).unsqueeze(1)
        stage_tok = self.stage_token.expand(B, -1, -1) + s

        # single cross-attention over band tokens
        q = stage_tok
        attn_out, _ = self.band_attn(q, tok, tok)

        # residual and feedforward
        y = self.norm0(attn_out + q)
        z = self.norm1(y + self.ff0(y))

        # squeeze to get final representation
        z = z.squeeze(1)

        # classifier output
        logits = self.head(z).squeeze(-1)
        return logits