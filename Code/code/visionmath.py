# author: Adrián Torremocha Doblas - Universidad de Málaga - 2023
import main_vm

while True:
    image_path = input("\nIntroduce the image name (STOP to exit): ")

    if image_path == "STOP":
        break

    show_image = input("Do you want to show the debug images? (True or False): ")

    debug = False
    if show_image == "True":
        debug = True
    elif show_image != "False":
        print("Incorrect value, please introduce True or False")
        continue

    main_vm.main(image_path, debug)

print('\nGoodbye! <3')
