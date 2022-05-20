import pandas as pd

heart = pd.read_csv(
    "/Users/noahbehling/ML_seminar/data/heart_2020_cleaned.csv", delimiter=","
)

heart["Diabetic"] = heart["Diabetic"].replace(
    {"No": 0, "No, borderline diabetes": 1, "Yes (during pregnancy)": 2, "Yes": 3}
)
heart["Sex"] = heart["Sex"].replace({"Female": 0, "Male": 1})
heart["Race"] = heart["Race"].replace(
    {
        "American Indian/Alaskan Native": 0,
        "Asian": 1,
        "Black": 2,
        "Hispanic": 3,
        "Other": 4,
        "White": 5,
    }
)
heart["AgeCategory"] = heart["AgeCategory"].replace(
    {
        "18-24": 0,
        "25-29": 1,
        "30-34": 2,
        "35-39": 3,
        "40-44": 4,
        "45-49": 5,
        "50-54": 6,
        "55-59": 7,
        "60-64": 8,
        "65-69": 9,
        "70-74": 10,
        "75-79": 11,
        "80 or older": 12,
    }
)

heart["GenHealth"] = heart["GenHealth"].replace(
    {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
)

heart = heart.replace({"No": 0, "Yes": 1})

heart_Female = heart.drop(heart[heart["Sex"] == 1].index)
heart_Male = heart.drop(heart[heart["Sex"] == 0].index)

heart_Female.to_csv("data/heart_2020_Female_encoded.csv", mode="wb")
heart_Male.to_csv("data/heart_2020_Male_encoded.csv", mode="wb")
