import gymnasium as gym
import gymnasium_robotics
import time


def run_fetch_reach():
    env = None
    try:
        # 加载机械臂环境
        env = gym.make("FetchReach-v3", render_mode="human")
        observation, info = env.reset(seed=42)

        print("机械臂环境加载成功")
        print(f"末端执行器初始位置：{observation['achieved_goal']}")
        print(f"目标位置：{observation['desired_goal']}\n")

        # 运行1500步
        for step in range(1500):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                print(f"第 {step + 1} 步：成功到达目标！")
                observation, info = env.reset()
            elif truncated:
                observation, info = env.reset()

            # 每100步打印一次进度
            if (step + 1) % 100 == 0:
                print(f"已完成 {step + 1}/1500 步")

        # 最后停留2秒观察
        time.sleep(2)

    except Exception as e:
        print(f"错误：{str(e)}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    run_fetch_reach()
