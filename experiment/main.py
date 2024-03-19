import argparse
from examples import ExampleSession
from feedback import FeedbackSession
from task import TaskSession

def main(subject, session, start_run, range, settings):

    example_session = ExampleSession()
    pass


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=str, help='Subject nr')
    argparser.add_argument('session', type=str, help='Session')
    argparser.add_argument('start_run', type=int, help='Run')
    argparser.add_argument('range', choices=['narrow', 'wide'], help='Range (either narrow or wide)')
    argparser.add_argument('--settings', type=str, help='Settings label', default='default')
    args = argparser.parse_args()
    main(args.subject, args.session, args.start_run, args.range, args.settings)