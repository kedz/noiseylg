import argparse
from pathlib import Path
import os
import random
from pprint import pprint
import json
from collections import OrderedDict


def handle_debug_opts(args, plum_pointers, checkpoints):

    if args.pprint_ds_sample is not None:
        pprint_sample_datasource(args.pprint_ds_sample,
                                 plum_pointers["datasources"], 
                                 args.pprint_ds_nsamples)

    if args.pprint_pipeline_sample is not None:
        pprint_sample_pipeline(args.pprint_pipeline_sample,
                               plum_pointers["pipelines"])

    if args.pprint_model is not None:
        pprint_model(args.pprint_model, plum_pointers["models"])

    if args.pprint_params is not None:
        pprint_params(args.pprint_params, plum_pointers["models"])

    if args.pprint_vocab is not None:
        pprint_vocab(args.pprint_vocab, plum_pointers["vocabs"])

def pprint_checkpoints(checkpoints):
    print("\nShowing saved checkpoints:")
    for ckpt, md in checkpoints.items():
        if md.get("default", False):
            print(" * {} | {}".format(ckpt, str(md["criterion"])))
        else:
            print("   {} | {}".format(ckpt, str(md["criterion"])))
    print("\n * default run is best checkpoint from latest run.")
    print()


def pprint_sample_datasource(datasource_names, datasources, num_samples):
    if datasource_names == []:
        datasource_names = list(datasources)
    
    for name in datasource_names:
        if name not in datasources:
            raise ValueError("No datasource with name: {}".format(name))
        datasource = datasources[name]

        print("Drawing {} samples from datasource {}".format(
            num_samples, name))
        indices = list(range(len(datasource)))
        random.shuffle(indices)
        sample_indices = indices[:num_samples]
        for n, idx in enumerate(sample_indices, 1):
            print(name, "sample {} no. {}".format(n, idx))
            pprint(datasource[idx])
        print()

def pprint_model(names, models):
    if names == []:
        names = list(models)

    print("Pretty printing models:\n")
    for name in names:
        print(name)
        pprint(models[name])
        print()
    print()

def pprint_sample_pipeline(pipeline_names, pipelines):
    if pipeline_names == []:
        pipeline_names = list(pipelines)

    for pipeline_name in pipeline_names:
        print(pipeline_name)
        for batch in pipelines[pipeline_name]:
            pprint(batch)
            break
        print()

def pprint_params(model_names, models):
    if model_names == []:
        model_names = list(models)

    for name in model_names:
        if name not in models:
            raise ValueError("No model with name: {}".format(name))
        model = models[name] 

        print("{} parameters:".format(name))

        names = []
        dtypes = []
        dims = []
        tags = []

        for pname, param in model.named_parameters():
            names.append(pname)
            dtypes.append(str(param.dtype))
            dims.append(str(tuple(param.size())))
            tags.append(str(model.parameter_tags(pname)))
    
        template = " {:" + str(max([len(x) for x in names])) + "s}" + \
            " | {:" + str(max([len(x) for x in dtypes])) + "s}" + \
            " | {:" + str(max([len(x) for x in dims])) + "s}" + \
            " | {:" + str(max([len(x) for x in tags])) + "s}" 

        for name, dtype, dim, tag in zip(names, dtypes, dims, tags):
            print(template.format(name, dtype, dim, tag))
        print()

def pprint_vocab(vocab_names, vocabs):
    if vocab_names == []:
        vocab_names = list(vocabs)

    for name in vocab_names:
        if name not in vocabs:
            raise ValueError("No vocab with name: {}".format(name))
        print(name)
        for idx, token in vocabs[name].enumerate():
            print(idx, token, vocabs[name].count(token))
        print() 

def get_meta_path():
    return Path.home() / ".plumr_meta.json"

def load_plumr_meta(verbose=False):
    meta_path = get_meta_path()

    if not meta_path.exists():
        meta_path.parent.mkdir(exist_ok=True, parents=True)
        meta_path.write_text(json.dumps({"ext_modules": []}))

    if verbose:
        print("Reading meta from: {}".format(meta_path))
    return json.loads(meta_path.read_text())

def update_ext_libs(meta, add_libs=None, del_libs=None, verbose=False):
    if add_libs is None:
        add_libs = []
    if del_libs is None:
        del_libs = []

    for lib in add_libs:
        if lib not in meta["ext_modules"]:
            try:
               __import__(lib)
               meta["ext_modules"].append(lib)
               if verbose:
                   print("Added lib: {}".format(lib))

            except Exception as e:
                print("Could not import: {}".format(lib))
                print("Got exception:")
                print(e)

    for lib in del_libs:
        if lib in meta["ext_modules"]:
            if verbose:
                print("Removing lib: {}".format(lib))
            meta["ext_modules"].pop(meta["ext_modules"].index(lib))

    get_meta_path().write_text(json.dumps(meta))

def import_ext_libs(meta):
    for lib in meta["ext_modules"]:
        __import__(lib)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--pprint", action="store_true")
    parser.add_argument("--pprint-ds-sample", default=None, nargs="*",
                        required=False)
    parser.add_argument("--pprint-ds-nsamples", default=4, type=int)
    parser.add_argument("--pprint-model", default=None, nargs="*",
                        required=False)
    parser.add_argument("--pprint-params", default=None, nargs="*",
                        required=False)
    parser.add_argument("--pprint-pipeline-sample", default=None, nargs="*",
                        required=False)
    parser.add_argument("--pprint-vocab", default=None, nargs="*",
                        required=False)
    parser.add_argument("--pprint-ckpts", action="store_true")
    parser.add_argument("--default-ckpt", type=str, default=None)
    parser.add_argument("-P", action="store_true")
    parser.add_argument("--run", type=str, nargs="+", default=None)
    parser.add_argument("--proj", type=Path, required=False, default=None)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--add-libs", nargs="+", default=None)
    parser.add_argument("--del-libs", nargs="+", default=None)
    args = parser.parse_args()

    pedantic = args.P

    if pedantic:
        print("\n ** Running in pedantic mode. Expect lots of messages. **\n")

    plumr_meta = load_plumr_meta(verbose=pedantic)
    update_ext_libs(plumr_meta, add_libs=args.add_libs, 
                    del_libs=args.del_libs, verbose=pedantic)
    if args.add_libs is not None or args.del_libs is not None:
        exit()
    
    if args.proj is None:
        project_directory = args.config.parent
    else:
        project_directory = args.proj

    import plum
    import_ext_libs(plumr_meta) 
 
    vocab_cache = project_directory / "vocabs"

    checkpoints = find_checkpoints(project_directory, args.default_ckpt)
    if args.pprint_ckpts:
        pprint_checkpoints(checkpoints)
 
    if args.config is None:
        return 
  
    if not args.config.exists():
        raise Exception("Config path doesn't exists: {}".format(args.config))

    plum_parser = plum.PlumParser(pprint_parse=args.pprint, 
                                  vocab_cache=vocab_cache,
                                  verbose=pedantic)
    plum_object, plum_pointers, config_json = plum_parser.parse_file(
        args.config, return_json=True)

    handle_debug_opts(args, plum_pointers, checkpoints)

    if args.run is not None:
        for program in args.run:
            if program not in plum_pointers["programs"]:
                raise RuntimeError(
                    "Program {} was not found in config {}".format(
                        program, args.config))
            else:
                env = create_environment(project_directory, program)
                env["checkpoints"] = checkpoints
                env["gpu"] = args.gpu
                (env["proj_dir"] / "config.json").write_text(config_json)
                plum_pointers["programs"][program].run(env, verbose=pedantic)


def find_checkpoints(root_dir, user_default):
    
    checkpoints = OrderedDict()
    default = None
    ckpt_metas = list(root_dir.rglob("ckpt.metadata.json"))
    ckpt_metas.sort(key=lambda x: os.stat(x).st_mtime)
    for path in ckpt_metas:
        
        meta = json.loads(path.read_text())
        
        default = ckpt_id = "{}:{}".format(
            path.parent.parent.name, meta["optimal_checkpoint"].split(".")[-2])

        for item in meta["checkpoint_manifest"][::-1]:
            ckpt_id = "{}:{}".format(
                path.parent.parent.name, item["checkpoint"].split(".")[-2])
            checkpoints[ckpt_id] = {
                "criterion": {meta["criterion"]: item["criterion"]},
                "path": path.parent / item["checkpoint"]
            }

    if len(checkpoints) == 0:
        return checkpoints

    if user_default is None:
        checkpoints[default]["default"] = True
    else:
        if user_default not in checkpoints:
            from warnings import warn
            warn("User upplied default checkpoint not found, using {}".format(
                default))

            checkpoints[default]["default"] = True
        else:
            checkpoints[user_default]["default"] = True
        
    return checkpoints
#?    for ckpt_id, md in checkpoints.items():
#?        for crit, val in md["criterion"].items():
#?            if ckpt_id == default:
#?                print(" * {} |  {} = {:6.7f}".format(ckpt_id, crit, val))
#?            else:
#?                print("   {} |  {} = {:6.7f}".format(ckpt_id, crit, val))

def create_environment(proj, prog):
    
#    ckpts = find_checkpoints(proj)
    proj_dir = proj / prog
    tb_dir = proj / "tb" / prog 

    run_num = 1
    run = "run1"
    while (proj_dir / run).exists() or (tb_dir / run).exists():
        run_num += 1
        run = "run{}".format(run_num)
    proj_dir = proj_dir / run
    tb_dir = tb_dir / run
    proj_dir.mkdir(exist_ok=True, parents=True)
    tb_dir.mkdir(exist_ok=True, parents=True)

    return {"proj_dir": proj_dir,   "tensorboard_dir": tb_dir}

if __name__ == "__main__":
    main()

