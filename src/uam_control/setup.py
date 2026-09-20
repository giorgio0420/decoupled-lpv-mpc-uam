from setuptools import find_packages, setup

package_name = "uam_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "osqp"],
    zip_safe=True,
    maintainer="Giorgio",
    maintainer_email="georgebest.gds@gmail.com",
    description=(
        "Decoupled dynamic modelling and tube-based LPV-MPC control for an "
        "aerial manipulator, after Eskandarpour et al., IEEE TAES 61(5), 2025."
    ),
    license="MIT",
)
