"""
This utility is called by Github Actions. It bumps the version number, unless the
patch part of the major.minor.patch version is 0. This would suggest a manual
major or minor release, in which case we probably don't want automatic patch 
increments.

It's expected that setup.py contains a call to setup, where one of the arguments
is the version number. This script rewrites setup.py to have a bumped version number.
Other Github Action workflows after it could commit this rewritten setup.py back to
the repository automatically.

The output of this script is always the bumped version number, which could be put
into an environment variable for later use in the workflow.
"""
import ast

import black

with open("setup.py") as f:
    parsed_setup = ast.parse(f.read())
    for element in parsed_setup.body:
        if isinstance(element, ast.Expr) and element.value.func.id == "setup":
            for keyword in element.value.keywords:
                if keyword.arg == "version":
                    original_version = keyword.value.value
                    major, minor, patch = keyword.value.value.split(".")
                    if patch == '0':
                        # If the last digit is 0, it suggests that a major or minor
                        # release was done manually. In that case, we add an extra 0.
                        # This is normalized by setuptools, so the release works just fine.
                        # The next commit to master will also work, because patch no longer
                        # is '0' but will be '00'. Quite hacky, but for now fixes this.
                        # Better ideas are welcome.
                        patch = '00'
                        bumped_version = keyword.value.value = f"{major}.{minor}.{patch}0"
                    else:
                        patch = str(int(patch) + 1)
                        bumped_version = keyword.value.value = f"{major}.{minor}.{patch}"
    bumped_setup = black.format_str(ast.unparse(parsed_setup), mode=black.FileMode())
    if patch == '00':
        print(bumped_version[:-1])
    else:
        print(bumped_version)

with open("setup.py", "w") as w:
    w.write(bumped_setup)
