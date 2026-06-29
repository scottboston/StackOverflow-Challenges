import pandas as pd

if __name__ == "__main__":

    scenarios = {
        "Scenario one": r"C:\Users\scott\PycharmProjects\StackOverflow-Challenges\data\challenge_20_s1.txt",
        "Scenario two": r"C:\Users\scott\PycharmProjects\StackOverflow-Challenges\data\challenge_20_s2.txt",
        "Scenario three": r"C:\Users\scott\PycharmProjects\StackOverflow-Challenges\data\challenge_20_s3.txt",
    }
    for scenario, file in scenarios.items():
        print(f"\n{scenario}:\n")
        df = pd.read_csv(
            file, sep=": ", engine="python", names=["name", "guests", "flavor"]
        )
        df_out = (
            df.groupby(by="flavor", sort=False)["guests"]
            .agg(["sum", "count"])
            .sum(axis=1)
            .mul(1.5)
            .floordiv(-9)
            .mul(-1)
        )
        for idx, value in df_out.items():
            print(f"  • {idx}: {value:.0f} tub{'s'[:int(value)^1]} ")
