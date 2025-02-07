### path setup ####################################################################################

import datetime

###################################################################################################
### Project Information ###########################################################################
###################################################################################################

project = "pulpo"
author = "Fabian Lechtenberg"
copyright = datetime.date.today().strftime("%Y") + " pulpo developers"
version: str = "latest"  # required by the version switcher

###################################################################################################
### Project Configuration #########################################################################
###################################################################################################

needs_sphinx = "7.3.0"

extensions = [
    # core extensions
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    # iPython extensions
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    # Markdown support
    # 'myst_parser', # do not enable separately if using myst_nb, compare: https://github.com/executablebooks/MyST-NB/issues/421#issuecomment-1164427544
    # Jupyter Notebook support
    "myst_nb",
    # mermaid support
    "sphinxcontrib.mermaid",
    # API documentation support
    "autoapi",
    # responsive web component support
    "sphinx_design",
    # custom 404 page
    "notfound.extension",
    # custom favicons
    "sphinx_favicon",
    # copy button on code blocks
    "sphinx_copybutton",
    # carousels
    "sphinx_carousel.carousel",
]

autoapi_dirs = ["../pulpo"]
autoapi_type = "python"
autoapi_ignore = [
    "*/data/*",
    "*tests/*",
    "*tests.py",
    "*validation.py",
    "*version.py",
    "*.rst",
    "*.yml",
    "*.md",
    "*.json",
    "*.data",
]

autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    #'special-members',
    #'imported-members',
    "show-inheritance-diagram",
]

autoapi_python_class_content = "both"
autoapi_member_order = "bysource"
autoapi_root = "content/api"
autoapi_template_dir = "_templates/autoapi_templates/"
autoapi_keep_files = False

graphviz_output_format = "svg"  # https://pydata-sphinx-theme.readthedocs.io/en/stable/examples/graphviz.html#inheritance-diagram

# Inject custom JavaScript to handle theme switching
mermaid_init_js = """
    function initializeMermaidBasedOnTheme() {
        const theme = document.documentElement.dataset.theme;

        if (theme === 'dark') {
            mermaid.initialize({
                startOnLoad: true,
                theme: 'base',
                themeVariables: {
                    edgeLabelBackground: 'transparent',
                    defaultLinkColor: '#ced6dd',
                    titleColor: '#ced6dd',
                    nodeTextColor: '#ced6dd',
                    lineColor: '#ced6dd',
                }
            });
        } else {
            mermaid.initialize({
                startOnLoad: true,
                theme: 'base',
                themeVariables: {
                    edgeLabelBackground: 'transparent',
                    defaultLinkColor: '#222832',
                    titleColor: '#222832',
                    nodeTextColor: '#222832',
                    lineColor: '#222832',
                }
            });
        }

        // Re-render all Mermaid diagrams
        mermaid.contentLoaded();
    }

    // Observer to detect changes to the data-theme attribute
    const themeObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
            initializeMermaidBasedOnTheme();
            }
        });
    });

    themeObserver.observe(document.documentElement, { attributes: true });

    initializeMermaidBasedOnTheme();
"""

master_doc = "index"

root_doc = "index"
html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"

suppress_warnings = [
    "myst.header"  # suppress warnings of the kind "WARNING: Non-consecutive header level increase; H1 to H3"
]


####################################################################################################
### Theme html Configuration #######################################################################
####################################################################################################

html_show_sphinx = False
html_show_copyright = True

html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",  # for https://fontawesome.com/ icons
]

html_sidebars = {
    "**": [
        "sidebar-nav-bs.html",
    ],
    "content/index": [],
    "content/installation": [],
    "content/theory": [],
    "content/contributing": [],
    "content/codeofconduct": [],
    "content/license": [],
    "content/changelog": [],
    "content/funding": [],
}

html_theme_options = {
    # page elements
    # "announcement": "⚠️ placeholder",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links.html"],
    "navbar_align": "content",
    # "navbar_persistent": ["theme-switcher"], # this is where the search button is usually placed
    "footer_start": ["copyright"],
    "footer_end": ["footer"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink", "support"],
    "header_links_before_dropdown": 5,
    # page elements content
    "icon_links": [
        {
            "name": "Open this Repo on GitHub",
            "url": "https://github.com/flechtenberg/pulpo",
            "icon": "fab fa-brands fa-github",
        },
    ],
    # various settings
    "collapse_navigation": True,
    # "show_prev_next": False,
    "use_edit_page_button": True,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/pulpo_logo_big.svg",
        "image_dark": "_static/pulpo_logo_big.svg",
    },
}

# required by html_theme_options: "use_edit_page_button"
html_context = {
    "github_user": "flechtenberg",
    "github_repo": "pulpo",
    "github_version": "master",
    "doc_path": "docs",
}

# notfound Configuration ################################################
# https://sphinx-notfound-page.readthedocs.io

notfound_context = {
    "title": "Page Not Found",
    "body": """                                                                                                                                           
        <h1>🍂 Page Not Found (404)</h1>
        <p>
        Oops! It looks like you've stumbled upon a page that's been recycled into the digital abyss.
        But don't worry, we're all about sustainability here.
        Why not take a moment to reduce, reuse, and recycle your clicks by heading back to the main page?
        And remember, every little bit counts in the grand scheme of things.
        </p>
    """,
}

####################################################################################################
### Extension Configuration ########################################################################
####################################################################################################

# myst_parser Configuration ############################################
# https://myst-parser.readthedocs.io/en/latest/configuration.html

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}


myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# myst-nb configuration ################################################
# https://myst-nb.readthedocs.io/en/latest/configuration.html

nb_execution_mode = "off"

# sphinx-favicon configuration #########################################
# https://github.com/tcmetzger/sphinx-favicon

favicons = [
    {"rel": "icon", "href": "favicon.svg", "type": "image/svg+xml"},
    {"rel": "icon", "sizes": "144x144", "href": "favicon-144.png", "type": "image/png"},
    {"rel": "mask-icon", "href": "favicon_mask-icon.svg", "color": "#222832"},
    {"rel": "apple-touch-icon", "sizes": "500x500", "href": "favicon-500.png"},
]
