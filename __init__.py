# -*- coding: UTF-8 -*-


from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .grag_nodes import GRAGPatch


class GRAG_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GRAGPatch]


async def comfy_entrypoint() -> GRAG_Extension:
    return GRAG_Extension()
