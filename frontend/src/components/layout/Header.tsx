import React, { useState, useRef, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  LogOut,
  User,
  ChevronDown,
  Coins,
  GraduationCap,
  School,
  Menu,
  X,
} from "lucide-react";
import { Button } from "../ui/Button";
import { useAuthContext } from "../../contexts/AuthContext";
import { useProfile } from "../../hooks/useProfile";

export function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, signOut } = useAuthContext();
  const { profile } = useProfile(user?.id);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const mobileMenuRef = useRef<HTMLDivElement>(null);

  const handleSignOut = async () => {
    await signOut();
    navigate("/auth");
    setMobileMenuOpen(false);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setDropdownOpen(false);
      }
      // Optional: Close mobile menu if clicking outside (though usually full width)
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
    setDropdownOpen(false);
  }, [location.pathname]);

  const isBlogPage = location.pathname.startsWith("/blog");

  return (
    <header className="sticky top-0 z-50 w-full border-b border-neutral-200 bg-white/80 backdrop-blur-md transition-all duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo Section */}
          <div className="flex items-center">
            <div
              className="flex items-center space-y-1 cursor-pointer group"
              onClick={() => navigate(user ? "/home" : "/")}
            >
              <div className="w-10 h-10 sm:w-12 sm:h-12 flex items-center justify-center transition-transform group-hover:scale-105">
                <img
                  src="/Evater_logo_2.png"
                  alt="Evater Logo"
                  className="w-full h-full object-contain"
                />
              </div>
              <div className="ml-2 hidden sm:block">
                <p className="text-lg text-neutral-900 font-bold tracking-tight">
                  Evater
                </p>
                <p className="text-xs text-neutral-500 font-medium -mt-1">
                  Next Gen Learning
                </p>
              </div>
            </div>
          </div>

          {/* Desktop Navigation & Actions */}
          <div className="hidden md:flex items-center space-x-4">
            {(!user || !isBlogPage) && (
              <Button
                onClick={() => navigate("/blog")}
                variant="ghost"
                size="sm"
                className="text-neutral-600 hover:text-primary-600 hover:bg-primary-50"
              >
                Blog
              </Button>
            )}
            <Button
              onClick={() => navigate("/about")}
              variant="ghost"
              size="sm"
              className="text-neutral-600 hover:text-primary-600 hover:bg-primary-50"
            >
              About
            </Button>

            {user ? (
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setDropdownOpen(!dropdownOpen)}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-full border transition-all duration-200 ${
                    dropdownOpen
                      ? "border-primary-200 bg-primary-50 ring-2 ring-primary-100"
                      : "border-neutral-200 hover:border-primary-200 hover:bg-neutral-50"
                  }`}
                >
                  <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center text-primary-700">
                    <User className="h-4 w-4" />
                  </div>
                  <span className="text-sm font-medium text-neutral-700 max-w-[100px] truncate">
                    {profile?.name || profile?.user_name || "User"}
                  </span>
                  <ChevronDown
                    className={`h-4 w-4 text-neutral-400 transition-transform duration-200 ${
                      dropdownOpen ? "rotate-180" : ""
                    }`}
                  />
                </button>

                {/* Desktop Dropdown */}
                {dropdownOpen && (
                  <div className="absolute right-0 mt-2 w-72 bg-white rounded-xl shadow-xl border border-neutral-100 ring-1 ring-black/5 overflow-hidden animate-in fade-in zoom-in-95 duration-100 origin-top-right">
                    <div className="p-4 border-b border-neutral-100 bg-neutral-50/50">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center text-primary-600">
                          <User className="h-5 w-5" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-semibold text-neutral-900 truncate">
                            {profile?.name || profile?.user_name || "User"}
                          </p>
                          <p className="text-xs text-neutral-500 truncate">
                            {user.email}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="p-2 space-y-1">
                      {/* Stats */}
                      <div className="px-3 py-2">
                        <div className="flex items-center justify-between text-sm mb-2">
                          <span className="text-neutral-500 flex items-center">
                            <GraduationCap className="w-3.5 h-3.5 mr-1.5" />{" "}
                            Grade
                          </span>
                          <span className="font-medium text-neutral-900">
                            {profile?.grade || "-"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm mb-2">
                          <span className="text-neutral-500 flex items-center">
                            <School className="w-3.5 h-3.5 mr-1.5" /> School
                          </span>
                          <span className="font-medium text-neutral-900 truncate max-w-[120px]">
                            {profile?.school || "-"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between p-2 bg-amber-50 rounded-lg border border-amber-100">
                          <span className="text-amber-700 text-sm flex items-center font-medium">
                            <Coins className="w-3.5 h-3.5 mr-1.5" /> Credits
                          </span>
                          <span className="font-bold text-amber-700">
                            {profile?.credits ?? "-"}
                          </span>
                        </div>
                      </div>

                      <div className="h-px bg-neutral-100 my-1" />

                      <button
                        onClick={() => navigate("/profile")}
                        className="w-full flex items-center px-3 py-2 text-sm text-neutral-600 hover:bg-neutral-50 hover:text-neutral-900 rounded-lg transition-colors"
                      >
                        Edit Profile
                      </button>
                      <button
                        onClick={handleSignOut}
                        className="w-full flex items-center px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      >
                        <LogOut className="w-4 h-4 mr-2" />
                        Sign Out
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center space-x-3">
                <Button
                  onClick={() => navigate("/auth")}
                  variant="ghost"
                  className="text-neutral-600 hover:text-neutral-900"
                >
                  Log in
                </Button>
                <Button
                  onClick={() => navigate("/auth")}
                  variant="primary"
                  className="shadow-md shadow-primary-500/20"
                >
                  Sign up
                </Button>
              </div>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-neutral-600 hover:bg-neutral-100 transition-colors"
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-neutral-200 bg-white shadow-lg animate-in slide-in-from-top-5 duration-200">
          <div className="px-4 py-6 space-y-4">
            {user ? (
              <>
                <div className="flex items-center p-4 bg-neutral-50 rounded-xl border border-neutral-100">
                  <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center text-primary-600 mr-4">
                    <User className="h-6 w-6" />
                  </div>
                  <div>
                    <p className="font-semibold text-neutral-900">
                      {profile?.name || "User"}
                    </p>
                    <p className="text-sm text-neutral-500">{user.email}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-neutral-50 rounded-lg border border-neutral-100">
                    <p className="text-xs text-neutral-500 mb-1">Grade</p>
                    <p className="font-medium text-neutral-900">
                      {profile?.grade || "-"}
                    </p>
                  </div>
                  <div className="p-3 bg-amber-50 rounded-lg border border-amber-100">
                    <p className="text-xs text-amber-600 mb-1">Credits</p>
                    <p className="font-bold text-amber-700">
                      {profile?.credits ?? "-"}
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <Button
                    onClick={() => navigate("/profile")}
                    variant="outline"
                    className="w-full justify-start"
                  >
                    Edit Profile
                  </Button>
                  <Button
                    onClick={handleSignOut}
                    variant="ghost"
                    className="w-full justify-start text-red-600 hover:bg-red-50 hover:text-red-700"
                  >
                    <LogOut className="w-4 h-4 mr-2" />
                    Sign Out
                  </Button>
                </div>
              </>
            ) : (
              <div className="space-y-3">
                <Button
                  onClick={() => navigate("/auth")}
                  variant="outline"
                  className="w-full justify-center"
                >
                  Log in
                </Button>
                <Button
                  onClick={() => navigate("/auth")}
                  variant="primary"
                  className="w-full justify-center"
                >
                  Sign up
                </Button>
              </div>
            )}
            <div className="pt-4 border-t border-neutral-100 space-y-2">
              {(!user || !isBlogPage) && (
                <Button
                  onClick={() => navigate("/blog")}
                  variant="ghost"
                  className="w-full justify-start"
                >
                  Blog
                </Button>
              )}
              <Button
                onClick={() => navigate("/about")}
                variant="ghost"
                className="w-full justify-start"
              >
                About
              </Button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
