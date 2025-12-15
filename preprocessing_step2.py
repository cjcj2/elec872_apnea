import numpy as np
from pathlib import Path
from scipy import signal as scipy_signal
from tqdm import tqdm
from typing import Tuple


class PreprocessMESA:
    def __init__(
        self,
        raw_fs: int = 256,
        target_fs: int = 128
    ):
        self.raw_fs = raw_fs
        self.target_fs = target_fs
        self.decim_factor = raw_fs // target_fs
        self.anti_alias_cutoff_hz = target_fs / 2.0

        # frequency bands
        self.bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
        self.band_names = ["delta", "theta", "alpha", "beta", "gamma"]

        # small epsilon for edge cases
        self._eps = 1e-6

    def _butter_lowpass(self, cutoff_hz: float, fs: int, order: int = 4):
        nyq = fs / 2.0
        wn = min(cutoff_hz / nyq, 1.0 - self._eps)
        return scipy_signal.butter(order, wn, btype="lowpass", output="sos")

    def _butter_bandpass(self, lowcut_hz: float, highcut_hz: float, fs: int, order: int = 4):
        nyq = fs / 2.0
        low = max(lowcut_hz / nyq, self._eps)
        high = min(highcut_hz / nyq, 1.0 - self._eps)

        return scipy_signal.butter(order, [low, high], btype="band", output="sos")

    def downsample_epoch(self, epoch: np.ndarray) -> np.ndarray:
        lp = self._butter_lowpass(self.anti_alias_cutoff_hz, fs=self.raw_fs, order=4)

        epoch_ds = []
        for ch in range(epoch.shape[0]):
            x = epoch[ch].astype(np.float64, copy=False)

            x_f = scipy_signal.sosfiltfilt(lp, x)
            x_r = scipy_signal.resample_poly(x_f, up=1, down=self.decim_factor)

            epoch_ds.append(x_r.astype(np.float32))

        return np.stack(epoch_ds, axis=0)

    def filter_bands(self, epoch_ds: np.ndarray) -> np.ndarray:
        n_channels, n_samples = epoch_ds.shape
        n_bands = len(self.band_names)
        bands = np.zeros((n_bands, n_channels, n_samples), dtype=np.float32)

        for band_idx, band_name in enumerate(self.band_names):
            lowcut, highcut = self.bands[band_name]

            sos_bp = self._butter_bandpass(lowcut, highcut, fs=self.target_fs, order=4)

            for ch in range(n_channels):
                x = epoch_ds[ch].astype(np.float64, copy=False)
                bands[band_idx, ch, :] = scipy_signal.sosfiltfilt(sos_bp, x).astype(np.float32)

        return bands

    def preprocess_file(self, input_path: Path, output_path: Path) -> Tuple[tuple, int]:
        data = np.load(input_path, allow_pickle=True)
        eeg = data["EEG"]
        stage = data["stage"]
        apnea = data["apnea"]

        n_epochs, n_channels, n_samples_raw = eeg.shape

        # downsampled length
        first_ds = self.downsample_epoch(eeg[0])
        n_samples_ds = first_ds.shape[-1]
        n_bands = len(self.band_names)

        eeg_bands = np.zeros((n_epochs, n_bands, n_channels, n_samples_ds), dtype=np.float32)

        for i in range(n_epochs):
            epoch_ds = self.downsample_epoch(eeg[i])
            eeg_bands[i] = self.filter_bands(epoch_ds)

        np.savez_compressed(
            output_path,
            eeg_bands=eeg_bands,
            stage=stage,
            apnea=apnea,
            fs=np.array(self.target_fs, dtype=np.int32),
            band_names=np.array(self.band_names),
            bands=np.array([self.bands[name] for name in self.band_names], dtype=np.float32),
        )

        return eeg_bands.shape, n_samples_ds


def preprocess_all_files(
    input_dir: str,
    output_dir: str,
    raw_fs: int = 256,
    target_fs: int = 128
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("mesa_*.npz"))

    preprocessor = PreprocessMESA(
        raw_fs=raw_fs,
        target_fs=target_fs
    )

    for input_path in tqdm(input_files, desc="preprocessing"):
        output_filename = input_path.stem + "_preprocessed.npz"
        output_path = output_dir / output_filename

        try:
            bands_shape, n_samples_ds = preprocessor.preprocess_file(input_path, output_path)
            print(f"{input_path.name}: eeg_bands={bands_shape}")
        except Exception as e:
            print(f"{input_path.name}: error - {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    preprocess_all_files(
        input_dir="/content/drive/MyDrive/ColabNotebooks/elec872/data/mesa_data_v4",
        output_dir="/content/drive/MyDrive/ColabNotebooks/elec872/data/mesa_data_v4_preprocessed",
        raw_fs=256,
        target_fs=128
    )

    print("\ncomplete")