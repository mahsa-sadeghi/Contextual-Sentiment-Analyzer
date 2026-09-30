from backend.cli.menu import show_menu
from backend.cli.router import handle_choice


def main():
    while True:
        show_menu()
        choice = input("\nChoose: ").strip()
        if not handle_choice(choice):
            break


if __name__ == "__main__":
    main()
