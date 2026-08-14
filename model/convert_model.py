# convert the .joblib model into JavaScript to use with React

import joblib
import m2cgen as m2c

model = joblib.load("model.joblib")
code = m2c.export_to_javascript(model)

with open("model.js", "w") as f:
    f.write(code)

print("Successfully generated model.js!")