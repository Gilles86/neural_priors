import os.path as op
from experiment.task import TaskSession

def main(settings='default'):
    settings_fn = op.join(op.dirname(__file__), 'settings', f'{settings}.yml')
    session = TaskSession(output_str='test', subject='test', output_dir='test', settings_file=settings_fn, 
                          run=1, eyetracker_on=False)

    print(session.settings)
    session.create_trials()
    session.run()


if __name__ == "__main__":
    main()
