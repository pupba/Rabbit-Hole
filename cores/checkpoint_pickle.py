"""
This file is part of ComfyUI.
Copyright (C) 2024 Comfy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pickle

load = pickle.load


class Empty:
    pass


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # TODO: safe unpickle
        if module.startswith("pytorch_lightning"):
            return Empty
        return super().find_class(module, name)
