from locust import LoadTestShape


class LoadTestPlan(LoadTestShape):
    stages = [
        {"duration": 20, "users": 1, "spawn_rate": 10},
        {"duration": 40, "users": 2, "spawn_rate": 10},
        {"duration": 60, "users": 4, "spawn_rate": 10},
        {"duration": 80, "users": 8, "spawn_rate": 100},
        {"duration": 100, "users": 16, "spawn_rate": 100},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                try:
                    tick_data = (stage["users"], stage["spawn_rate"], stage["user_classes"])
                except:
                    tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None
