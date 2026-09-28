"""Run streaming speaker diarization on audio files and write RTTM.

Usage:
    python -m fastdiar.run_diarizer audio.wav                      # -> audio.rttm next to it
    python -m fastdiar.run_diarizer audio.wav -o out/              # -> out/audio.rttm
    python -m fastdiar.run_diarizer audio.wav -o out/result.rttm
    python -m fastdiar.run_diarizer data/ --ext .flac -o rttm/ -m small

A directory input is searched recursively for files with the given extension.
Without ``--output`` every RTTM is saved next to its audio file; with it, the
RTTM files go to that directory, mirroring the input's subdirectories.
"""

from tqdm import tqdm

from fastdiar.cli import build_parser, load_audio, load_model, plan_jobs
from fastdiar.diarizer import StreamingDiarizer


def main() -> None:
    args = build_parser(__doc__, output_help="RTTM").parse_args()
    jobs = plan_jobs(args.input, args.output, args.ext, suffix=".rttm")
    diarizer = StreamingDiarizer(load_model(args))

    for audio_path, rttm_path in tqdm(jobs, unit="file", disable=not args.input.is_dir()):
        # RTTM fields are space-separated, so the file id cannot contain spaces.
        rttm = diarizer(load_audio(audio_path), uri=audio_path.stem.replace(" ", "_"))
        rttm_path.parent.mkdir(parents=True, exist_ok=True)
        rttm_path.write_text(rttm)


if __name__ == "__main__":
    main()
