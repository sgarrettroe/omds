import yaml
from scidatalib.scidata import SciData
import json

file_name = 'demo/demo-experiment-description.yaml'


def process_yaml(in_file_name):
    """
    Process an input yaml file and make a jsonld file
    :param in_file_name: name of the yaml file to load
    :return sci_data_obj: A SciData instance
    """
    stream = open(in_file_name, 'r')
    yaml_content = yaml.load(stream, Loader=yaml.CLoader)
    print(yaml_content)

    # create instance with unique id
    sci_data_obj = SciData(yaml_content['uid'])

    sci_data_obj.context(yaml_content['context'])

    # what does base mean?
    sci_data_obj.base(yaml_content['base'])

    sci_data_obj.namespaces(yaml_content['namespaces'],replace=False)

    # what is the difference between uid and docid?
    sci_data_obj.docid(yaml_content['docid'])

    sci_data_obj.title(yaml_content['title'])

    sci_data_obj.author(yaml_content['authors'])

    sci_data_obj.description(yaml_content['description'])

    sci_data_obj.publisher(yaml_content['publisher'])

    sci_data_obj.keywords(yaml_content['keywords'])

    # this is a problem -- we haven't made the file, so how do I know it's permanent location. Update later?
    sci_data_obj.permalink(yaml_content['link'])

    sci_data_obj.discipline(yaml_content['discipline'])

    sci_data_obj.subdiscipline(yaml_content['subdiscipline'])

    # add the type of methodology - experimental, computational, etc.
    # I don't understand how fine / coarse this should be
    sci_data_obj.evaluation(yaml_content['evaluation'])

    # how to get instrument settings from the data?
    # what is the relationship between aspects and data sets?
    sci_data_obj.aspects(yaml_content['aspects'])

    sci_data_obj.facets(yaml_content['facets'])

    return sci_data_obj


sci_data_obj = process_yaml(file_name)
print(json.dumps(sci_data_obj.output, indent=4, ensure_ascii=False))
