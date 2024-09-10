def check_compatibility():
    try:
        import mistralai
        import pydantic
        print(f"Mistral AI version: {mistralai.__version__}")
        print(f"Pydantic version: {pydantic.__version__}")
        return True
    except ImportError as e:
        print(f"Error de importación: {e}")
        return False
    except TypeError as e:
        print(f"Error de tipo: {e}")
        return False

if __name__ == "__main__":
    if not check_compatibility():
        print("El entorno no es compatible. Por favor, revisa las versiones de las dependencias.")
        # Aquí puedes decidir si continuar con funcionalidad limitada o detener la ejecución
    else:
        print("El entorno es compatible.")