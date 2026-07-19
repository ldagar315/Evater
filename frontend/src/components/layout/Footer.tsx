import React from "react";
import { Link } from "react-router-dom";
import { Heart, Mail } from "lucide-react";
import { useAuthContext } from "../../contexts/AuthContext";

const footerLinks = {
  product: [
    { label: "Home", to: "/home" },
    { label: "Practice", to: "/practice" },
    { label: "Leaderboard", to: "/leaderboard" },
    { label: "Profile", to: "/profile" },
  ],
  resources: [
    { label: "Blog", to: "/blog" },
    { label: "Study guides", to: "/blog?category=study-skills" },
    { label: "About Evater", to: "/about" },
    { label: "Contact support", to: "mailto:hello@evater.com" },
  ],
  legal: [
    { label: "Privacy overview", to: "/about#privacy" },
    { label: "Terms overview", to: "/about#terms" },
    { label: "Cookie overview", to: "/about#cookies" },
  ],
};

export function Footer() {
  const { user } = useAuthContext();
  const currentYear = new Date().getFullYear();
  const productLinks = footerLinks.product.map((link) => ({
    ...link,
    to: user ? link.to : link.label === "Home" ? "/" : "/auth",
  }));

  return (
    <footer className="mt-auto border-t border-neutral-200 bg-neutral-50 pb-8 pt-12 sm:pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-10 grid grid-cols-1 gap-10 md:grid-cols-2 lg:mb-12 lg:grid-cols-4 lg:gap-12">
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
              A focused practice space for building confidence, understanding
              mistakes, and learning with your peers.
            </p>
            <div className="flex items-center gap-3 pt-2">
              <a
                href="mailto:hello@evater.com"
                className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-neutral-200 bg-white px-3 text-sm font-bold text-neutral-600 transition-colors hover:border-primary-200 hover:text-primary-700"
                aria-label="Email Evater support"
              >
                <Mail className="h-4 w-4" aria-hidden="true" />
                Email Evater
              </a>
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3 className="font-semibold text-neutral-900 mb-4">Product</h3>
            <ul className="space-y-3">
              {productLinks.map((link) => (
                <li key={link.label}>
                  <Link
                    to={link.to}
                    className="inline-flex min-h-11 items-center text-sm text-neutral-600 hover:text-primary-600 transition-colors"
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
                  {link.to.startsWith("mailto:") ? (
                    <a href={link.to} className="inline-flex min-h-11 items-center text-sm text-neutral-600 transition-colors hover:text-primary-600">
                      {link.label}
                    </a>
                  ) : (
                    <Link to={link.to} className="inline-flex min-h-11 items-center text-sm text-neutral-600 transition-colors hover:text-primary-600">
                      {link.label}
                    </Link>
                  )}
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
                    className="inline-flex min-h-11 items-center text-sm text-neutral-600 hover:text-primary-600 transition-colors"
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
