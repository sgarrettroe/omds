OMGS: 0001
Title: Pioneering Open Sceince with OpenMDS
Author: Shawna Sinchak shs355@pitt.edu
Sponsor: Sean Garrett-Roe sgr@pitt.edu
Discussions-To: https://github.com/sgarrettroe/omds
Status: Draft
Type: Informational
Topic: Governance
Created: 10-03-2025
Python-Version: 3.14

Abstract
========

Open-Multidimensional spectroscopy (OpenMDS) aims to provide an interface to improve
the depth of reporting research data and results to promote machine-readability and
comprehensive record keeping for the multidimensional spectroscopy community.
As a central repository, OpenMDS operates through a standard for richly describing
the context of data.
With the richly described data in one location, the organization, storage, and access
of large data sets can become more efficient.
According to the fourth paradigm of scientific discovery, large collections of uniformly
annotated data encourage scientific discovery [#para]_ .
A set of data standards and an openly accessible platform for communication in the
multidimensional spectroscopy community opens opportunities for noticing new experimental
trends, new collaborations, and verifying experiments more easily.

Open Science encourages free access of scientific data and sharing resources to
make the community as a whole better.
To achieve a system available to researchers, national labs,
and industry, the data submitted should follow FAIR and TRUE
(Transparent, Reproducible, Usable for others, Extensible) attributes.
The full context of the experiment can be described with metadata files in the
SciData Framework [#sci]_ .
The Scidata Framework provides an outline on how to provide relevant information to describe
the background and significance of data in relation to the experimental goals and
conditions.
By annotating the experimental data with the context of the experiment and
adding the information to a central repository, OpenMDS can serve as a valuable
and reliable resource for sharing, accessing, and analyzing multidimensional
spectroscopy data.


Motivation
==========

Known as the fourth paradigm of scientific discovery, large repositories of richly
described data provide more in-depth and accelerated analysis of scientific
results [#para]_ .
The internet provides a platform for scientists to aggregate large, diverse
datasets and to communicate with other scientists worldwide.
By implementing infrastructure that makes data reusable and understandable,
scientific communities can experience the same increase of efficiency
consumer companies experienced with the mobilization of the semantic web [#para]_ .
The semantic web relates the context of data across websites, making data unique
and universal [#semWebOnto]_ .
Currently, there are no platforms in the multidimensional spectroscopy community
that facilitate open communication for data sharing, integrating machine learning,
or reproducing experiments across different labs.
Here, OpenMDS will bridge the gap between large aggregates of data
and making that data accessible to many audiences.

The practice of a central repository with richly described data proves successful
in the context of the Protein Data Bank (PDB).
Created in the 1970s with only 13 protein residues, the PDB has now (50 years later)
expanded to over 200,000 unique proteins with in-depth structural descriptions [#pbd]_ .
Development and training of computational platforms, such as AlphaFold2
and its derivatives, is possible as a direct result of the PDB [#AF]_ .
AlphaFold2 thoroughly tests protein binding events, so clinical trials
have a higher success rate, saving time and money in the field of pharmaceuticals.
Additionally, the central repository allows for expedited worldwide collaboration,
conformation of experimental results, and pattern recognition.
As an analogous platform to the PDB for multidimensional spectroscopy,
OpenMDS could enhance collaborations, data analysis, and reproduction of results.

As seen with the PDB, the principle of Open Science -- promoting transparent
and descriptive dissemination of research procedures and results -- aids in the
development of effective large repositories by fostering community curation
and a flexible repository.
To accomplish this, the data stored in the repository follows FAIR
and TRUE data standards.

FAIR defines the standards for data to be Findable, Accessible, Interoperable,
and Reusable [#fair]_ .
Findable data standards pair the raw data with richly annotated files to describe the
broader context and uniqueness of the data.
Accessible data standards protect the data with authorization points when necessary and
maintain a log of the repository history.
Interoperable data standards integrate the data into automated workflows
and machine learning.
Finally, reusable data standards adhere to protocols for creativity license and
how data can be used as a reference in other works.
By tailoring the infrastructure of OpenMDS to FAIR data standards,
OpenMDS can be a reliable resource for accessing, storing, and analyzing data.

TRUE standards aim to create software that is transparent, reproducible,
usable by others, and extensible [#true]_ .
The transparency and reproducibility allows for more intuitive repurposing
and revising of code.
The usable by others and extensible aspects of the software allow
for improvements.
Through following TRUE standards, OpenMDS acts as a flexible
document as the purpose and reach of the software evolves.

To tie it all together, FAIR and TRUE standards can be achieved through
including files that contain data about the data, known as metadata files [#semWebOnto]_ .
These metadata files operate through a similar mechanism as the internet
with a network of connected metadata known as the semantic web [#semWebOnto]_ .
The internet works through Uniform Resource Locators, familiarly known as
URLs, to direct the user to desired information.
A downfall of this system is that URLs do not quantify the content on
webpages, so Uniform Resource Identifiers (URIs) were created to make
the information visible on the semantic web [#semWebOnto]_ .
URIs are unique, allowing for the information to be easily updated
across platforms and machine readable.
The URIs are contained within metadata files.
For OpenMDS, we primarily use JavaScript Object Notation for Linked Data (JSON-LD)
files.
Since the semantic web and metadata are existing methods to make data
unique and easily found, these tools will help encourage scientific data
accessibility within OpenMDS and allow others to contribute to OpenMDS.

The SciData framework provides a flexible format for describing the metadata
that coincides with research data [#sciData]_ .
Derived from the scientific method, the SciData framework identifies the who
and what of the experiment, the instrumentation details, the experimental set up,
and the data acquired from the experiment [#sci]_ .
Specifically, SciData formats metadata for scientific data in 5 general
categories of provenance, methodology, aspects, facets, and dataset [#sci]_ .
Within these broad categories, attributes unique to the experiment can be
added as necessary while still being machine readable and following community
standards.

A central repository in the multidimensional spectroscopy community provides
a solution to communicating and accessing data efficiently across domains.
Improved communication facilitates verifying data via experimental or
computational methods, building the credibility of performed experiments.
Additionally, with the ability to analyze others' data more efficiently,
more opportunities of inspiration for collaborations or future research will arise.
Furthermore, richly annotated experimental data allows for better data analysis
across different experiments and minimizes redundancy within individual labs.
Considering the impact a central repository has on interpreting and sharing data,
implementing and maintaining OpenMDS streamlines the spectroscopist's
workflow.


Rationale
=========

Every component within the SciData framework works towards to purpose of
providing a formal definition of the meaning of metadata and data, so
it can be understood by the general collective.
By embodying the principles of Open Science and FAIR data, the flexible SciData
framework fits perfectly into the needs of OpenMDS.
Since members of the multidimensional spectroscopy community have their own labs
and instruments with different research goals and parameters, each lab
communicates their data in different standards.
The SciData framework provides a medium for scientists to thoroughly describe
research parameters in the JSON-LD files in a way that makes the data comparable
and surrounded by a specific context [#sci]_ .
Other attempts to make data accessible have been made, but the software was
constructed in isolation, preventing the global community from
benefiting from it [#sci]_ .
Additionally, if the attempts were made within specific communities,
the software may lack extensibility or depend too much on the
specific implementation.
The previously mentioned downfalls also contribute to data incompatibility
with machine learning techniques.
Rigid semantic data prevents flexibility for incorporating different purposes.
Similarly, the lack of semantic data prevents generating accessible context
for the data at hand.
The SciData framework for JSON-LD files overcomes these previous barriers
by not defining the data required [#sci]_ .
Instead, the SciData framework outlines relevant metadata containers for
researchers needs that are customizable based on the researcher's purpose.

OpenMDS sets out to agglomerate various types of experimental data and organize
it to aid in comparing and finding the data long term.
Thus, choosing the SciData framework provides a method to relate scientific data
and to place it in the context of the semantic web.
According to the World Wide Web Consortium (w3) that defines the protocols for
communicating information on the internet, Linked Data requires that it is
associated with a URI and uses a link that follows the secure Hypertext
Transfer Protocol (HTTPs) [#w3]_ .
To make each piece of Linked Data findable, other URIs and information contained
within a standard format accompany the original piece of data [#w3]_ .
Even though the linked data from the experiment is placed into a machine readable internet
standard, the SciData framework and JSON-LD file format help communicate the semantic data in a
human readable fashion.
Additionally, SciData applies the extensibility and flexibility of graph databases
(which situates data in a network of nodes and edges), but maintains the
specificity of relational databases (storing data with key-value pairs) [#sci]_ .
The SciData Framework achieves this through including containers within the
file that nest within other containers.
A more thorough description of this formating is provided in the specifications.
While JSON-LD files human readable, future plans for the project include
developing software that accompanies online scientific notebooks to integrate
the metadata collection process into the already existing workflow of the
scientist.

Not only SQL database languages, such as SHACL (Shapes Constraint Language)
and SPARQL (SPARQL protocol and RDF Query Language) interpret the data of JSON-LD files.
These languages verify the containers within the file and provide a platform
for researchers to retrieve their data from OpenMDS.
The SciData framework also defines the references and copyright stipulations
within the metadata [#sci]_ .
These two categories provide proper credit for the research collected and
allows for the researcher to limit the permissions of their data.


Specification
=============


The semantic web is the internet standard to frame information in
a way that an automated process can understand [#semWebOnto]_ .
The semantic web attaches Uniform Resource Identifier (URI) to pieces
of information using ontologies.
Ontologies are databases for specific fields of knowledge or professions
that associate terms with their URIs as a reference.
Agreeing on how to identify each term eliminates confusion and ambiguity
within the community.
This is useful if something is known by several names or if two names
are similar.
To make data compatible with the semantic web, it should be contained in
a file in a standard format, contain globally unique identifiers,
and describe itself [#sci]_ .
Additionally, since computers depend on explicit definitions to distinguish
items, richly annotated data helps develop machine learning processes
and databases.

The semantic web acts as a network connecting URIs together, making the web
flexible to new additions of information.
The semantic web operates through the resource description framework (RDF)
and RDF triples.
RDF triples relate each piece of metadata to itself and others, creating
a web interconnecting the points.
RDF triples link two data points with a descriptor that define their
relationship with each other [#semWebOnto]_ .
Thus, RDF triples organize metadata into a sentence structure of
subject, predicate, and object [#CIT2022]_ .
Since each piece of metadata can act as a subject or object with respect
to other pieces of metadata, a network with thorough and explicit relations
can be generated.
A machine can then survey this network and determine the unique qualities of
each metadata piece efficiently.
The data is findable, accessible, and interoperable, making it machine readable.


.. figure:: docs/img.png

   Figure 1. Graphical representation of simplified RDF triple web [#CIT2022]_ .


A metadata file describes RDF triples.
For this repository, a JavaScript object notation for
linked data (JSON-LD) file contains the metadata.
The JSON-LD file is both human readable and machine readable, making it abide
by the principles of FAIR data.
JSON-LD files incorporate unique identifiers and relate the data to each other,
making the data findable and accessible in a repository [#sciData]_ .
The SciData framework formats the metadata within the JSON-LD file to
also aid in human readability and experimental flexibility.

The SciData framework contains the provenances and then outlines the
scientific process with a methodology, system and dataset section.
Since it is organized into general bins that the scientist can customize,
this blueprint allows for easy editing depending on the purpose and how
much data is available.

.. figure:: docs/img_2.png

  Figure 2. Sample excerpt of SciData JSON-LD file [#sci]_ .

The JSON-LD file contains the Scidata Object and the context of the
file.
Thus, at the root level of the file are contained the attributions and
the provenance to describe the information contained within the 'SciData'
bin.
To provide more specifics of the metadata suggested at the root level,
the authors, author information, title, publisher, license, and keywords
are included in this section.
Ontologies for referencing the unique identifiers of metadata points
within the SciData container are also defined.
Within the root of the file is an identifier labeled 'toc,' which instructs
the file where to find other hierarchical components within the JSON-LD file.
In the scope of SciData, the 'toc' identifier will point to the 'aspects',
'facets' and 'dataset' [#sci]_ .
The 'aspects' refers to the experimental methodology followed when collecting
the data.
The 'facets' refers to the system or systems that are being observed.
Lastly, the 'dataset' contains the reported elements of data for the experiment.

To explore the 'Scidata' container more, each sub container of 'aspects',
'facets' and 'dataset' can contain their own specific kinds of hierarchies.
Each sub container can contain a list of properties for the reported data.
The 'aspect' or methodology describes how the research was obtained.
The methodology includes but is not limited to the instrumentation,
parameters, procedure, and software used.
The 'facets' or system subcategory provides context on the species of interest within
the experiment.
The compounds, purities, mixtures, and lab conditions are explained,
and are organized within another hierarchy to describe each one.

The bin labeled 'dataset' reports the data in the SciData framework.
Knowing that data can be reported in different contexts and relate to each other,
there are three different ways to characterize data within the bin of 'dataset'.
These include 'datapoint', 'dataseries' and 'datagroup'.
Defining an individual point of data, 'datapoint' is the smalled level of
data identification.
The 'datapoint' signification implies the point can stand alone or that
there is no relation to the other experimental values included within the
'dataset' bin.
The next bin that experimental values can placed into is the 'dataseries'.
The 'dataseries' bin connects logically related pieces of data to each other.
Some examples of information that 'dataseries' include varying a specific property,
such as time, or a spectrum of data.
The defining feature of the 'dataseries' is that it included correlated data
or multiple correlated arrays.
The 'dataseries' can also be defined as a json array or a json array of internal
links to previously defined 'datapoints' within the same file.
The largest component of the data organization heiarchiny within the SciData
framework is the 'datagroup'.
The 'datagroup' connects experimental values that have a higher level connection
to each other.
The 'datagroup' is the most flexible bin within the 'dataset' container since
it can contain other 'datagroups' without a limit, 'dataseries' and 'datapoints'.
The flexibility of the 'datagroup' allows for the aggregation of data based on the
scientists needs [#sci]_ .

The hierarchies within the 'dataset' container gain significance when they are related
to information in the 'aspects' or 'facets' containers.
Through the structure of the JSON-LD file, different points of the file can be related to
each other.
JSON-LD files contain JSON-LD context, which are identifiers that can be used to reference
the meanings of each JSON-LD name value pairs.
A JSON-LD context is created by the keyword @context, and it generates a unique identifier for
the parameter and its value.
The unique identifier is provided from a URI from an ontology.
The created JSON-LD context can then be referenced within the file to connect it with
an specific definition signaled by an @id tag [#sciData]_.
To further describe the data, @type defines the data type of the object included [#sciData]_ .

With this set up, the SciData framework follows an open and easily searchable format,
and it is flexible.

Footnotes
=========


.. [#para]  Hey, T., Tansley, S., Tolle, K., & Gray, J. (2009). The Fourth Paradigm: Data-Intensive Scientific Discovery. Microsoft Research. https://www.microsoft.com/en-us/research/publication/fourth-paradigm-data-intensive-scientific-discovery/

.. [#sci] Chalk, S. J. (2016). SciData: a data model and ontology for semantic representation of scientific data. Journal of Cheminformatics, 8(1), 54. https://doi.org/10.1186/s13321-016-0168-9

.. [#semWebOnto] Allemang, D., Hendler, J., & Gandon, F. (2020). Semantic Web for the Working Ontologist. ACM. https://doi.org/10.1145/3382097

.. [#pbd]  Berman, H. M. (2008). The Protein Data Bank: a historical perspective. Acta Crystallographica Section A, 64(1), 88–95. https://doi.org/10.1107/S0108767307035623

.. [#AF] Yang, Z., Zeng, X., Zhao, Y., & Chen, R. (2023). AlphaFold2 and its applications in the fields of biology and medicine. Signal Transduction and Targeted Therapy, 8(1), 115. https://doi.org/10.1038/s41392-023-01381-z

.. [#fair] https://www.go-fair.org/fair-principles/

.. [#true] https://www.tandfonline.com/doi/full/10.1080/00268976.2020.1742938

.. [#sciData] https://stuchalk.github.io/scidata/

.. [#w3] https://www.w3.org/TR/ldp/

.. [#CIT2022] https://www.w3.org/RDF/Metalog/docs/sw-easy


Copyright
=========

GNU General Public License v3.0