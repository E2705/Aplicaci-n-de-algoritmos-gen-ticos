import sys
from src import MaximizacionCubica, OptimizaciónMultivariable, ImpactoPM, ElitismoAmpliado


def mostrar_menu() -> None:
    """
    Esta función tiene como objetivo mostrar el menú 
    de opciones disponibles en el programa por medio
    de prints para que se muestren en la consola.
    """

    print("\n   Actividad Algoritmos Geneticos (ML)")
    print("--------------------------------------------------")
    print("\n Presentado por Erik Arevalo y Oscar Duque")
    print("\n1. Ejercicio 1: Maximización Cubica")
    print("2. Ejercicio 2: Optimización Multivariable en el Cromosoma")
    print("3. Ejercicio 3: Impacto en la Tasa de Mutación")
    print("4. Ejercicio 4: Elitismo de 3 Individuos")
    print("5. Salir")

def main() -> None:
    """
    La función "def main() -> None" va a ejecutar un ciclo while true
    el cual solo se ve interrumpido si el usuario decide salir o llamar a un ejercicio.
    """
    while True:
        mostrar_menu()
        opcion = input("¿Que ejercicio desea ejecutar?: ").strip()

        if opcion == "1":
            MaximizacionCubica.ejecutar()
        elif opcion == "2":
            OptimizaciónMultivariable.ejecutar()
        elif opcion == "3":
            ImpactoPM.ejecutar()
        elif opcion == "4":
            ElitismoAmpliado.ejecutar()
        elif opcion == "5":
            print("\nFin del programa")
            sys.exit(0)
        else:
            print("\nEsta opción no existe. Por favor ingrese un número del 1 al 5.")

if __name__ == "__main__":
    main()