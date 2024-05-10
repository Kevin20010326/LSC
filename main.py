# main.py

# 導入數據預處理和模型定義的模塊
from data_preprocessing import prepare_data
from model_training import create_model

def train_and_evaluate():
    # 數據預處理
    train_data, test_data = prepare_data()

    # 創建模型
    model = create_model()

    # 訓練模型
    model.fit(train_data)

    # 在測試集上進行評估
    evaluation_result = model.evaluate(test_data)

    return evaluation_result

def main():
    # 訓練並評估模型
    evaluation_result = train_and_evaluate()

    # 打印評估結果
    print("Evaluation result:", evaluation_result)

if __name__ == "__main__":
    main()
