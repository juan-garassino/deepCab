from deepCab.interface.main import preprocess, train, evaluate

from prefect import task, Flow, Parameter

import os
import requests


@task
def preprocess_new_data(experiment):
    """
    Run the preprocessing of the new data
    """
    preprocess()
    preprocess(source_type="val")


@task
def evaluate_production_model(status):
    """
    Run the `Production` stage evaluation on new data
    Returns `eval_mae`
    """
    eval_mae = evaluate()
    return eval_mae


@task
def re_train(status):
    """
    Run the training
    Returns train_mae
    """
    train_mae = train()
    return train_mae


@task
def notify(eval_mae, train_mae):
    base_url = "https://wagon-chat.herokuapp.com"
    channel = "johnini"
    url = f"{base_url}/{channel}/messages"
    author = "johnini"
    content = "Evaluation MAE: {} - New training MAE: {}".format(
        round(eval_mae, 2), round(train_mae, 2)
    )
    data = dict(author=author, content=content)
    response = requests.post(url, data=data)
    response.raise_for_status()


"""@task
def eval_perf(next_row):

    # evaluate latest production model on new data
    past_perf = evaluate()

    print(Fore.GREEN + "\n🔥 Ran task: EVAL PERF:" + Style.RESET_ALL +
          f"\n- Past model performance: {past_perf}")

    return past_perf


@task
def train_model(next_row):

    # preprocess data chunk by chunk
    preprocess()
    preprocess(source_type="val")

    # train model chunk by chunk
    new_perf = train()

    print(Fore.GREEN + "\n🔥 Ran task: TRAIN MODEL:" + Style.RESET_ALL +
          f"\n- New model performance: {new_perf}")

    return new_perf


@task
def notify(past_perf, new_perf):

    print(Fore.GREEN + f"\n🔥 Run task: NOTIF" + Style.RESET_ALL +
          f"\n- Past performance: {past_perf}" +
          f"\n- New performance: {new_perf}")

def build_flow(schedule):

    with Flow(name="garassino workflow", schedule=schedule) as flow:

        next_row = 0

        # evaluate the performance of the past model
        past_perf = eval_perf(next_row)

        # retrain the model with new lines
        new_perf = train_model(next_row)

        # print results
        notify(past_perf, new_perf)

    return flow


if __name__ == "__main__":

    # schedule = None
    schedule = IntervalSchedule(interval=datetime.timedelta(minutes=2),
                                end_date=datetime.datetime(2022, 11, 25))

    flow = build_flow(schedule)

    # flow.visualize()

    # flow.run()

    flow.executor = LocalDaskExecutor()

    flow.register("garassinojuan project")
"""


def build_flow():
    """
    build the prefect workflow for the `taxifare` package
    """
    flow_name = os.environ.get("PREFECT_FLOW_NAME")

    with Flow(flow_name) as flow:

        # retrieve mlfow env params
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")

        # create workflow parameters
        experiment = Parameter(name="experiment", default=mlflow_experiment)

        # register tasks in the workflow
        status = preprocess_new_data(experiment)
        eval_mae = evaluate_production_model(status)
        train_mae = re_train(status)
        notify(eval_mae, train_mae)

    return flow
