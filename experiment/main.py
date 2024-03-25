import argparse
from examples import ExampleSession
from feedback import FeedbackSession
from task import TaskSession
from utils import get_output_dir_str, get_settings
import os.path as op
from score import ScoreSession

def main(subject, session, start_run, stimulus_range, settings, n_runs=None,
         run_examples=True, run_feedback=True):

    run = start_run

    if n_runs is None:
        n_runs = 4

    output_dir, output_str = get_output_dir_str(subject, session, 'examples', run=run)

    settings_fn, use_eyetracker = get_settings(settings)


    sessions = []

    if run_examples:
        example_session = ExampleSession(output_str=output_str,
                                        subject=subject,
                                        output_dir=output_dir,
                                        settings_file=settings_fn,
                                        run=run,
                                        range=stimulus_range,
                                        eyetracker_on=use_eyetracker) 
        example_session.create_trials()
        example_session.run()


    if run_feedback:
        output_dir, output_str = get_output_dir_str(subject, session, 'feedback', run=run)
        feedback_session = FeedbackSession(output_str=output_str,
                                        subject=subject,
                                        output_dir=output_dir,
                                        settings_file=settings_fn,
                                        run=run,
                                        range=stimulus_range,
                                        eyetracker_on=use_eyetracker)

        feedback_session.create_trials()
        feedback_session.run()
    
    for run in range(start_run, start_run+n_runs):
        output_dir, output_str = get_output_dir_str(subject, session, 'estimation_task', run=run)
        task_session = TaskSession(output_str=output_str, subject=subject,
                        output_dir=output_dir, settings_file=settings_fn, 
                        run=run, range=stimulus_range, eyetracker_on=use_eyetracker)

        task_session.create_trials()
        task_session.run()

    output_dir, output_str = get_output_dir_str(subject, session, 'score', run=0)

    score_session = ScoreSession(output_str=output_str,
                            range='narrow',
                            subject=subject, output_dir=output_dir, eyetracker_on=use_eyetracker,
                            settings_file=settings_fn)

    score_session.create_trials(session)
    score_session.run()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=str, help='Subject nr')
    argparser.add_argument('session', type=str, help='Session')
    argparser.add_argument('start_run', type=int, help='Run')
    argparser.add_argument('range', choices=['narrow', 'wide'], help='Range (either narrow or wide)')
    argparser.add_argument('--settings', type=str, help='Settings label', default='default')
    argparser.add_argument('--n_runs', type=int, default=4, help='n_runs_to_run') 
    argparser.add_argument('--no_examples', action='store_false', help='Do not run examples block')
    argparser.add_argument('--no_feedback', action='store_false', help='Do not run feedback block')
    args = argparser.parse_args()
    main(args.subject, args.session, args.start_run, args.range, args.settings, args.n_runs,
         run_examples=args.no_examples, run_feedback=args.no_feedback)