from backend.cli.analysis import analyze_sentence
from backend.cli.word_analysis import show_negative_words, show_positive_words
from evaluation.evaluate import run_evaluation


def handle_choice(choice):
    if choice == "1":
        analyze_sentence()
    elif choice == "2":
        show_positive_words()
    elif choice == "3":
        show_negative_words()
    elif choice == "4":
        report = run_evaluation()
        print("\nEvaluation")
        print("----------")
        print(f"Sentiment accuracy: {report['accuracy']}%")
        print(f"Sentiment macro F1: {report['macro_f1']}%")
        print(f"Topic accuracy: {report['topic']['accuracy']}%")
    elif choice == "5":
        return False
    else:
        print("Invalid option")
    return True
