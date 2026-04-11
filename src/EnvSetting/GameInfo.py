import platform
import datetime

def GameSetting(configlist, ENVname):
    date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    team0, team1 = configlist[0], configlist[1]
    team0_frame, team1_frame, team0_alg = team0.Framework.upper(), team1.Framework.upper(), team0.Algorithm
    load_path = {0:"", 1:""}
    if team1_frame=='RSA':
        env_path = f"RSA/{team0_frame}/{ENVname}/{team0_frame}vsRSA"
        save_path = f"results/{team0_frame}vsRSA/{ENVname}/{team0_alg}/{date_time[4:8]}/{date_time[8:]}"
        load_path[0] = configlist[0].load_path

        # HRL이면 ppo 로드할 수 있는 부분도 만들어야함

        print(f"[Environment Setting] {ENVname} : {team0_frame}-{team0_alg} vs {team1_frame}")
        print(f"Load Model / Train: {team0.load_model} / {team0.train_mode}")
    elif team1_frame=='RBA':
        env_path = f"RBA/{ENVname}/{team0_frame}vsRBA"
        save_path = f"results/{team0_frame}vsRBA/{ENVname}/{team0_alg}/{date_time[4:8]}/{date_time[8:]}"
        load_path[0] = configlist[0].load_path

        print(f"[Environment Setting] {ENVname} : {team0_frame}-{team0_alg} vs {team1_frame}")
        print(f"Load Model / Train: {team0.load_model} / {team0.train_mode}")
    else:
        team1_alg = configlist[1].Algorithm
        load_path[0] = configlist[0].load_path
        load_path[1] = configlist[1].load_path
        game = f"{team0_frame}vs{team1_frame}"
        # 게임 이름이라던가 여러가지 하이퍼 파라미터는 hyperparameter.py만 수정하시면 됩니다
        save_path = f"results/{game}/{ENVname}/{team0_alg}vs{team1_alg}/{date_time[4:8]}/{date_time[8:]}"
        env_path = f"{game}/{ENVname}/{game}"
        print(f"[Environment Setting] {ENVname} : {team0_frame}-{team0_alg} vs {team1_frame}-{team1_alg}")
        print(f"Load Model - Train : {team0.load_model} - {team0.train_mode} | {team1.load_model} - {team1.train_mode}")
    os_name = platform.system()

    # env_name : UnityBuild.exe file path
    if os_name == 'Windows':
        env_name = f"UnityEnv/{os_name}/{env_path}.exe"
    elif os_name =='Linux':
        env_name = f"UnityEnv/{os_name}/{env_path}.x86_64"
    elif os_name == 'Darwin': # macOS
        env_name = f"../UnityEnv/{game}/{game}_{os_name}"
    else:
        raise RuntimeError("Not Supported OS")

    return env_name, save_path, load_path
