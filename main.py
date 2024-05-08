from front import create_front


def main() -> None:
    try:
        create_front()
    except Exception as error:
        print(error)


if __name__ == '__main__':
    main()
