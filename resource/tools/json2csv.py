# -*- coding: utf-8 -*-


import pandas as pd
import argparse
import json


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = config()
    sessions = json.load(open(args.json_file))
    sessions_tmp = []
    for session in sessions:
        session_idx = session["session_idx"]
        session = session["session"]
        session_tmp = []
        for episode in session:
            episode["session_idx"] = session_idx
            session_tmp.append(episode)
        sessions_tmp += session_tmp
    df = pd.DataFrame(sessions_tmp)
    title = ["session_idx", "turn_idx", "post", "gth", "state", "action", "hyp"]
    df = df[title]
    df.to_excel(args.csv_file)


if __name__ == '__main__':
    main()
