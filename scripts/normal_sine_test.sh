DATA=mono_normal
CONSTRAINT=sine
python main-train.py config/train/$DATA.py
python main-generate.py config/constrained_generation/$CONSTRAINT.py