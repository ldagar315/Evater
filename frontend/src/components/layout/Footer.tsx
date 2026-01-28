import React from "react";
import { Link } from "react-router-dom";
import { Twitter, Github, Linkedin, Mail, Heart } from "lucide-react";

const footerLinks = {
  product: [
    { label: "Home", to: "/home" },
    { label: "Create Test", to: "/create-test" },
    { label: "Previous Tests", to: "/previous-tests" },
    { label: "Previous Feedbacks", to: "/previous-feedbacks" },
  ],
  resources: [
    { label: "Blog", to: "/blog" },
    { label: "Help Center", to: "#" },
    { label: "Guides", to: "#" },
    { label: "API Status", to: "#" },
  ],
  legal: [
    { label: "Privacy Policy", to: "#" },
    { label: "Terms of Service", to: "#" },
    { label: "Cookie Policy", to: "#" },
  ],
};

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-neutral-50 border-t border-neutral-200 pt-16 pb-8 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
          {/* Brand Column */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <img
                src="/Evater_logo_2.png"
                alt="Evater Logo"
                className="w-10 h-10 object-contain"
              />
              <span className="text-xl font-bold text-neutral-900">Evater</span>
            </div>
            <p className="text-neutral-600 text-sm leading-relaxed">
              Empowering educators and students with AI-driven assessments.
              Create, share, and analyze tests with ease.
            </p>
            <div className="flex space-x-4 pt-2">
              <a
                href="#"
                className="text-neutral-400 hover:text-primary-600 transition-colors"
              >
                <Twitter className="h-5 w-5" />
              </a>
              <a
                href="#"
                className="text-neutral-400 hover:text-primary-600 transition-colors"
              >
                <Github className="h-5 w-5" />
              </a>
              <a
                href="#"
                className="text-neutral-400 hover:text-primary-600 transition-colors"
              >
                <Linkedin className="h-5 w-5" />
              </a>
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3 className="font-semibold text-neutral-900 mb-4">Product</h3>
            <ul className="space-y-3">
              {footerLinks.product.map((link) => (
                <li key={link.label}>
                  <Link
                    to={link.to}
                    className="text-sm text-neutral-600 hover:text-primary-600 transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources Links */}
          <div>
            <h3 className="font-semibold text-neutral-900 mb-4">Resources</h3>
            <ul className="space-y-3">
              {footerLinks.resources.map((link) => (
                <li key={link.label}>
                  <Link
                    to={link.to}
                    className="text-sm text-neutral-600 hover:text-primary-600 transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal Links */}
          <div>
            <h3 className="font-semibold text-neutral-900 mb-4">Legal</h3>
            <ul className="space-y-3">
              {footerLinks.legal.map((link) => (
                <li key={link.label}>
                  <Link
                    to={link.to}
                    className="text-sm text-neutral-600 hover:text-primary-600 transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-neutral-200 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-xs text-neutral-500">
            © {currentYear} Evater. All rights reserved.
          </p>
          <div className="flex items-center text-xs text-neutral-500">
            <span>Made with</span>
            <Heart className="h-3 w-3 mx-1 text-red-500 fill-current" />
            <span>for better learning</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
