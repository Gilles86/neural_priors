import glob
import os.path as op
import argparse
import pandas as pd
import re
from session import EstimationSession
from instruction import InstructionTrial
from utils import get_output_dir_str, get_settings
from exptools2.core import PylinkEyetrackerSession, Trial


def get_subject_stats(subject, session, log_dir, max_reward=.1, reward_slope=.025, no_response_penalty=0.1):
    feedback_log_files = glob.glob(op.join(log_dir, f'sub-{subject}', f'ses-{session}', '*task-feedback_run-*_events.tsv'))
    estimation_log_files = glob.glob(op.join(log_dir, f'sub-{subject}', f'ses-{session}', '*task-estimation_task_run-*_events.tsv'))
    print(op.join(log_dir, f'sub-{subject}', f'ses-{session}', '*task-feedback_run-*_events.tsv'))

    reg = re.compile('.*/sub-(?P<subject>.+)_ses-(?P<session>[0-9]+)_task-(?P<task>[a-zA-Z_]+)_run-(?P<run>[0-9]+)_events.tsv')

    stats = {}

    feedback_df = []
    for fn in feedback_log_files:
        run = reg.match(fn).group('run')
        d = pd.read_csv(fn, sep='\t')
        d['run'] = run
        feedback_df.append(d)

    feedback_df = pd.concat(feedback_df).set_index(['run', 'trial_nr', 'event_type']).xs('feedback', level='event_type').astype({'n':float, 'response':float})
    feedback_error = feedback_df['n'] - feedback_df['response']

    stats['mean_error_feedback'] = feedback_error.mean()
    stats['mean_abs_error_feedback'] = feedback_error.abs().mean()
    stats['mean_squared_error_feedback'] = feedback_error.pow(2).mean()
    stats['n_no_responses_feedback'] = feedback_error.isnull().sum()
    stats['total_reward_feedback'] = stats['n_no_responses_feedback'] *  no_response_penalty
    feedback_error = feedback_error[~feedback_error.isnull()]
    stats['total_reward_feedback'] += (max_reward - feedback_error.pow(2) * reward_slope).sum()
    stats['total_n_feedback_trials'] = len(feedback_error)

    estimation_df = []

    for fn in estimation_log_files:
        run = reg.match(fn).group('run')
        d = pd.read_csv(fn, sep='\t')
        d['run'] = run
        estimation_df.append(d)

    estimation_df = pd.concat(estimation_df).set_index(['run', 'trial_nr', 'event_type']).xs('feedback', level='event_type').astype({'n':float, 'response':float})
    estimation_error = estimation_df['n'] - estimation_df['response']

    stats['mean_error_estimation'] = estimation_error.mean()
    stats['mean_abs_error_estimation'] = estimation_error.abs().mean()
    stats['mean_squared_error_estimation'] = estimation_error.pow(2).mean()
    stats['n_no_responses_estimation'] = estimation_error.isnull().sum()
    stats['total_reward_estimation'] = stats['n_no_responses_estimation'] *  no_response_penalty
    estimation_error = estimation_error[~estimation_error.isnull()]
    stats['total_reward_estimation'] += (max_reward - estimation_error.pow(2) * reward_slope).sum()
    stats['total_n_estimation_trials'] = len(estimation_error)

    return stats

    
class ScoreSession(EstimationSession):

    def create_trials(self, session):

        no_response_penalty = self.settings['score']['no_response_penalty']
        max_reward = self.settings['score']['max_reward']
        reward_slope = self.settings['score']['reward_slope']

        log_dir = op.join(op.dirname(__file__), 'logs')
        stats = get_subject_stats(self.settings['subject'], session, log_dir, max_reward, reward_slope, no_response_penalty)

        txt1 = f"During the feedback task, you performed {stats['total_n_feedback_trials']} trials.\n\n" \
            f"On {stats['n_no_responses_feedback']} trials, you did not respond in time, and you were penalized with {stats['n_no_responses_feedback'] *  no_response_penalty:.2f} CHF.\n\n" \
            f"You made an average error of {stats['mean_error_feedback']:.2f} and an average absolute error of {stats['mean_abs_error_feedback']:.2f}.\n\n" \
            f"In total, you earned {stats['total_reward_feedback']:.2f} CHF for the blocks with feedback."

        txt2 = f"During the estimation task, you performed {stats['total_n_estimation_trials']} trials.\n\n" \
            f"On {stats['n_no_responses_estimation']} trials, you did not respond in time, and you were penalized with {stats['n_no_responses_estimation'] *  no_response_penalty:.2f} CHF.\n\n" \
            f"You made an average error of {stats['mean_error_estimation']:.2f} and an average absolute error of {stats['mean_abs_error_estimation']:.2f}.\n\n" \
            f"In total, you earned {stats['total_reward_estimation']:.2f} CHF for the blocks with feedback."
        
        txt3 = f"On top of your hourly fee, you will get a bonus of {stats['total_reward_feedback'] + stats['total_reward_estimation']:.2f} CHF for the whole session."

        self.trials = [InstructionTrial(self, 0, txt1), InstructionTrial(self, 0, txt2), InstructionTrial(self, 0, txt3)]

        

def main(subject, session, settings):

    output_dir, output_str = get_output_dir_str(subject, session, 'score', 0)

    settings_fn, use_eyetracker = get_settings(settings)

    session_ = ScoreSession(output_str=output_str,
                            range='narrow',
                           subject=subject, output_dir=output_dir, eyetracker_on=use_eyetracker,
                           settings_file=settings_fn)

    session_.create_trials(session)

    session_.run()

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=str, help='Subject nr')
    argparser.add_argument('session', type=str, help='Session')
    argparser.add_argument('--settings', type=str, help='Settings label', default='default')
    args = argparser.parse_args()

    main(args.subject, args.session, args.settings)

