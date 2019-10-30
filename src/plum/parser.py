import _jsonnet
try:
    import ujson as json
except ModuleNotFoundError:
    import json
from pathlib import Path
import hashlib

try: 
    from importlib.resources import read_text
except ModuleNotFoundError:
    from importlib_resources import read_text

from pprint import pprint

from .types import PLUM_OBJECT_REGISTRY
import plum


class PlumParser:
    def __init__(self, registry=None, pprint_parse=False,
                 vocab_cache=None, verbose=False):
        if registry is None:
            registry = PLUM_OBJECT_REGISTRY
        self._registry = registry
        self._pprint_parse = pprint_parse
        self._verbose = verbose
        self.vocab_cache = vocab_cache

    @property
    def pprint_parse(self):
        return self._pprint_parse

    @property
    def verbose(self):
        return self._verbose

    def _import_callback(self, dir, rel):
        regular_path = Path(dir) / rel
        if regular_path.exists():
            return str(regular_path), regular_path.read_text()

        else:
            return "plum/jsonnet/" + rel, read_text("plum.jsonnet", rel)
            # RuntimeError(rel not found)

    def parse_string(self, config_string, return_json=False):

        if self.verbose:
            print("Plum Parser: parsing string")
        
        json_string = _jsonnet.evaluate_snippet(
            "snippet", config_string, import_callback=self._import_callback)
        config = json.loads(json_string)

        if self.pprint_parse:
            print("Parsed Config:")
            pprint(config)
            print()

        obj, ptrs = self._build_config(config)

        if return_json:
            return obj, ptrs, config_string
        else:
            return obj, ptrs

    def parse_file(self, config_path, return_json=False):

        if self.verbose:
            print("Plum Parser: parsing file {}".format(config_path))

        config_string = _jsonnet.evaluate_file(
            str(config_path), import_callback=self._import_callback)
        config = json.loads(config_string)

        if self.pprint_parse:
            print("Parsed Config:")
            pprint(config)
            print()

        obj, ptrs = self._build_config(config)

        if return_json:
            return obj, ptrs, config_string
        else:
            return obj, ptrs

    def _build_config(self, config):

        pointers = {"datasources": dict(), "vocabs": dict(), "models": dict(),
                "programs": dict(), "pipelines": dict()}
        plum_obj = self._recurse_and_parse(config, pointers)

        if self.verbose:
            print("Plum Parser: parsed the following objects")

            if len(pointers["datasources"]):
                print("\n [datasources]")
                for name in pointers["datasources"]:
                    print("  ", name)

            if len(pointers["vocabs"]):
                print("\n [vocabs]")
                for name in pointers["vocabs"]:
                    print("  ", name)

            if len(pointers["pipelines"]):
                print("\n [pipelines]")
                for name in pointers["pipelines"]:
                    print("  ", name)

            if len(pointers["models"]):
                print("\n [models]")
                for name in pointers["models"]:
                    print("  ", name)

            if len(pointers["programs"]):
                print("\n [programs]")
                for name in pointers["programs"]:
                    print("  ", name)
            print()

        return plum_obj, pointers

    def _construct_plum_type(self, plum_id, arg_dict):
        return self._registry.new_object(plum_id, arg_dict)

    def _recurse_and_parse(self, config_item, pointers):

        if isinstance(config_item, (list, tuple)):
            return [self._recurse_and_parse(x, pointers) for x in config_item]
            
        elif not isinstance(config_item, dict):
            return config_item

        else:
            plum_keys = {}
            for key in list(config_item):
                value = config_item[key]
                if key.startswith("__plum_"):
                    plum_keys[key] = value
                    del config_item[key]

            canon_json = json.dumps(config_item, sort_keys=True) 

            for key in list(config_item):
                value = config_item[key]
                config_item[key] = self._recurse_and_parse(
                    value, pointers)

            if "__plum_model__" in plum_keys:

                model_name = plum_keys["__plum_model__"]
                if model_name in pointers["models"]:
                    return pointers["models"][model_name]

                if "__plum_model_load__" in plum_keys:
                    model = plum.load(plum_keys["__plum_model_load__"])
                    pointers["models"][model_name] = model
                    return model

                model_type = plum_keys.get("__plum_type__", None)
                if model_type is None:
                    raise Exception(
                        "Model {} missing type()".format(model_name))

                plum_obj = self._construct_plum_type(model_type, config_item)
                pointers["models"][model_name] = plum_obj
                return plum_obj

            elif "__plum_vocab__" in plum_keys:
                return self.construct_vocab(plum_keys, config_item, canon_json,
                                            pointers["vocabs"])

            elif "__plum_datasource__" in plum_keys:

                ds_name = plum_keys["__plum_datasource__"]
                if ds_name in pointers["datasources"]:
                    return pointers["datasources"][ds_name]
                else:
                    obj_type = plum_keys.get("__plum_type__", None)
                    if obj_type is None:
                        raise Exception("Object missing type()")

                    plum_obj = self._construct_plum_type(obj_type, config_item)
                    pointers["datasources"][ds_name] = plum_obj
                    return plum_obj

            elif "__plum_pipeline__" in plum_keys:

                pipe_name = plum_keys["__plum_pipeline__"]
                if pipe_name in pointers["pipelines"]:
                    return pointers["pipelines"][pipe_name]
                    #raise Exception("pipeline names must be unqiue: {}".format(
                    #    pipe_name))
                else:
                    obj_type = plum_keys.get("__plum_type__", None)
                    if obj_type is None:
                        raise Exception("Object missing type()")

                    plum_obj = self._construct_plum_type(obj_type, config_item)
                    pointers["pipelines"][pipe_name] = plum_obj
                    return plum_obj

            elif "__plum_program__" in plum_keys:

                pm_name = plum_keys["__plum_program__"]
                if pm_name in pointers["programs"]:
                    raise Exception("Program names must be unique!")
                else:
                    obj_type = plum_keys.get("__plum_type__", None)
                    if obj_type is None:
                        raise Exception("Program missing type()")

                    plum_obj = self._construct_plum_type(obj_type, config_item)
                    pointers["programs"][pm_name] = plum_obj
                    return plum_obj
            elif "__plum_vocab_op__" in plum_keys:
                func_type = plum_keys.get("__plum_type__", None)
                args = [self._recurse_and_parse(x, pointers) 
                        for x in plum_keys["__plum_vocab_op__"]]
                return self._registry._registry[func_type](*args)

            elif "__plum_type__" in plum_keys: 
                obj_type = plum_keys.get("__plum_type__", None)
                if obj_type is None:
                    raise Exception("Object missing type()")

                plum_obj = self._construct_plum_type(obj_type, config_item)
                return plum_obj

            else:
                return {key: self._recurse_and_parse(value, pointers)
                        for key, value in config_item.items()} 

    def construct_vocab(self, plum_keys, vocab_args, canonical_json, pointers):

        name = plum_keys["__plum_vocab__"]

        # Vocab has already been constructed, return that instance from 
        # pointers dict.
        if name in pointers:
            return pointers[name]

        obj_type = plum_keys.get("__plum_type__", None)
        if obj_type is None:
            raise Exception("Plum vocab '{}' missing __plum_type__".format(
                name))

        # No vocab caching directory is set, so just construct vocab from the
        # supplied args and return it. 
        if self.vocab_cache is None \
                or plum_keys.get("__plum_vocab_no_cache__", False):
            vocab = self._construct_plum_type(obj_type, vocab_args)
            pointers[name] = vocab
            return vocab

        # Caching is enabled so generate md5 hash for vocab args and underlying
        # data source paths/last modified times. 
        string_buffers = []
        if hasattr(vocab_args["dataset"], "paths"):
            for path in vocab_args["dataset"].paths:
                string_buffers.append(str(Path(path).resolve()))
                string_buffers.append(str(Path(path).stat().st_mtime))
        else:
            for obj in vocab_args["dataset"]:
                string_buffers.append(str(obj))

        string_repr = canonical_json + "\n" + "  ".join(string_buffers)
        m = hashlib.md5()
        m.update(string_repr.encode("utf8"))
        md5 = m.hexdigest()

        self.vocab_cache.mkdir(exist_ok=True, parents=True)
        vocab_meta = self.vocab_cache / "{}.meta".format(name)
        cache_path = self.vocab_cache / "{}.pth".format(name)

        if not vocab_meta.exists() or vocab_meta.read_text() != md5 or \
                not cache_path.exists():

            if self.verbose:
                print("building vocab: {}".format(name))

            vocab = self._construct_plum_type(obj_type, vocab_args)
            vocab.save(cache_path)
            pointers[name] = vocab
            vocab_meta.write_text(md5)
            return vocab

        else:

            if self.verbose:
                print("reading cached vocab '{}' from {}".format(
                    name, cache_path))
            vocab = plum.load(cache_path)
            pointers[name] = vocab
            return vocab


        pointers[name] = vocab

        
        #print(canonical_json)
        #print(obj_type)
        #print(obj_configs)
        exit()
#                return self.construct_vocab(plum_keys, config_item, canon_json,
#                                            pointer["vocabs"])
#                                      
#
#                        raise Exception("Object missing type()")
#                    print(obj_type, config_item)
#                    exit()
#                    plum_obj = self._construct_plum_type(obj_type, config_item)
#                    pointers["vocabs"][v_name] = plum_obj
#                    return plum_obj


