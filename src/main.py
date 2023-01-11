import sys
import os
from src.azure.azure_connection import AzureConnection


def main():
    os.chdir("./azure")
    azure_con = AzureConnection()
    azure_con.run_script()


if __name__ == "__main__":
    main()